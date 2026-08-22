// Instance-lifetime, host-free caches for pairwise links and accumulated fields.
//
// The cache deliberately contains no work callback.  A caller captures a generation, does
// frame pulls/inference/composition without this object, and then conditionally publishes the
// result.  That shape is important: clipGetImage and an estimator run can both re-enter host or
// runtime code, and holding this mutex across either operation would make a deadlock possible.

#ifndef WHITEWATER_CORE_FLOW_FLOWCACHE_H
#define WHITEWATER_CORE_FLOW_FLOWCACHE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "core/flow/FlowLink.h"

namespace whitewater {

// Keep the schema in the key even while the cache is RAM-only.  It makes an entry's meaning
// explicit and leaves room for a future user-managed durable cache without changing the key
// contract.  This is a cache schema version, not a model-choice or bake-off index.
inline constexpr std::uint32_t kFlowCacheSchemaVersion = 1;

// Pairwise and accumulated entries use the same correctness identity.  The two stores remain
// distinct because their values have different reuse semantics and eviction pressure.
struct FlowCacheKey {
  int fromFrame = 0;
  int toFrame = 0;
  std::uint64_t generation = 0;

  // Opaque, stable tokens.  The model token is normally the model file's SHA-256.  The other
  // strings describe mathematical input conditioning and model parameters without naming a
  // Phase 2.5 choice or assigning an API index to one.
  std::string modelFingerprint;
  std::string matteToken;
  std::string inputConditioningToken;
  std::string modelParametersFingerprint;

  // A field lattice is identified by its dimensions and both exact endpoint geometries. The
  // geometries are values rather than derived scales: PAR and independently rounded X/Y
  // analysis pitches must not alias.
  int modelColumns = 0;
  int modelRows = 0;
  FieldGeometry modelGeometry;
  FieldGeometry destinationGeometry;

  std::uint32_t schemaVersion = kFlowCacheSchemaVersion;

  bool operator==(const FlowCacheKey &other) const;
  bool operator!=(const FlowCacheKey &other) const { return !(*this == other); }
};

struct FlowCacheKeyHash {
  std::size_t operator()(const FlowCacheKey &key) const noexcept;
};

// Values hold immutable shared objects.  A render may retain a hit after the cache evicts it;
// shared ownership keeps that result alive without extending the cache's byte accounting.
struct PairwiseFlowValue {
  std::shared_ptr<const FlowLink> link;
  std::shared_ptr<const ScalarField> confidence;

  PairwiseFlowValue() = default;
  explicit PairwiseFlowValue(std::shared_ptr<const FlowLink> linkValue,
                             std::shared_ptr<const ScalarField> confidenceValue = nullptr)
      : link(std::move(linkValue)), confidence(std::move(confidenceValue)) {}

  // Copying into shared storage is useful for short-lived preparation results and keeps the
  // cache API symmetric with AccumulatedFieldValue.  The hot path should pass shared_ptrs.
  explicit PairwiseFlowValue(const FlowLink &linkValue,
                             std::shared_ptr<const ScalarField> confidenceValue = nullptr);

  bool isValid() const { return link != nullptr && link->isValid(); }
  std::size_t byteSize() const;
};

struct AccumulatedFieldValue {
  std::shared_ptr<const FlowLink> link;
  std::shared_ptr<const ScalarField> confidence;

  AccumulatedFieldValue() = default;
  explicit AccumulatedFieldValue(std::shared_ptr<const FlowLink> linkValue,
                                 std::shared_ptr<const ScalarField> confidenceValue = nullptr)
      : link(std::move(linkValue)), confidence(std::move(confidenceValue)) {}

  explicit AccumulatedFieldValue(const FlowLink &linkValue,
                                 std::shared_ptr<const ScalarField> confidenceValue = nullptr);

  bool isValid() const { return link != nullptr && link->isValid(); }
  const FlowField &field() const { return link->field(); }
  std::size_t byteSize() const;
};

struct FlowCacheBudgets {
  std::size_t pairwiseBytes = 0;
  std::size_t accumulatedBytes = 0;
};

class FlowCache {
 public:
  using Generation = std::uint64_t;

  // A distinct token prevents accidentally passing an arbitrary frame or cache-schema number to
  // conditional publication.  The value is public so callers can copy it into FlowCacheKey.
  struct GenerationToken {
    Generation value = 0;

    bool operator==(const GenerationToken &other) const { return value == other.value; }
    bool operator!=(const GenerationToken &other) const { return !(*this == other); }
  };

  FlowCache();
  explicit FlowCache(FlowCacheBudgets budgets);

  // Convenience form for callers that want the same per-store ceiling.  Production code can
  // use FlowCacheBudgets to divide one artist-facing budget explicitly between the stores.
  explicit FlowCache(std::size_t perStoreByteBudget);
  FlowCache(std::size_t pairwiseByteBudget, std::size_t accumulatedByteBudget);

  ~FlowCache();
  FlowCache(FlowCache &&) noexcept;
  FlowCache &operator=(FlowCache &&) noexcept;
  FlowCache(const FlowCache &) = delete;
  FlowCache &operator=(const FlowCache &) = delete;

  Generation generation() const;
  GenerationToken captureGeneration() const;

  // Bumping invalidates every entry, including entries that happen to have matching non-generation
  // fields.  This is the correctness boundary for changedClip and flow-affecting parameters.
  Generation bumpGeneration();
  Generation invalidate() { return bumpGeneration(); }

  // Drop resident values and advance the generation. This is the operation behind Clear;
  // advancing prevents work captured before the button press from repopulating the cache.
  void clear();

  // Budget changes evict least-recently-used values immediately.  A zero budget is a supported,
  // disabled cache; it contains no entries.  An entry larger than its store's budget is not kept.
  void setBudgets(FlowCacheBudgets budgets);
  void setPairwiseByteBudget(std::size_t byteBudget);
  void setAccumulatedByteBudget(std::size_t byteBudget);

  FlowCacheBudgets budgets() const;
  std::size_t pairwiseByteCount() const;
  std::size_t accumulatedByteCount() const;
  std::size_t totalByteCount() const;
  std::size_t pairwiseSize() const;
  std::size_t accumulatedSize() const;

  // A hit updates recency.  The returned shared value remains valid if another thread later
  // clears or evicts the cache.
  std::optional<PairwiseFlowValue> lookupPairwise(const FlowCacheKey &key);
  std::optional<AccumulatedFieldValue> lookupAccumulated(const FlowCacheKey &key);

  // These inspection methods intentionally do not update recency.  They are useful to planning
  // code that wants to avoid disturbing eviction order and to deterministic policy tests.
  bool containsPairwise(const FlowCacheKey &key) const;
  bool containsAccumulated(const FlowCacheKey &key) const;

  // Direct insertion is for already-computed values (for example, a precache walk).  It accepts
  // only the current generation.  Oversize, invalid, and disabled entries remove any prior value
  // under the same key and return false.
  bool putPairwise(const FlowCacheKey &key, PairwiseFlowValue value);
  bool putAccumulated(const FlowCacheKey &key, AccumulatedFieldValue value);

  // Conditional publication is the render path's stale-result guard.  The value is sized before
  // entering the cache mutex; no caller work or user callback executes while the mutex is held.
  // Publication also requires key.generation == token.value.
  bool publishPairwise(GenerationToken token, const FlowCacheKey &key,
                       PairwiseFlowValue value);
  bool publishAccumulated(GenerationToken token, const FlowCacheKey &key,
                          AccumulatedFieldValue value);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FLOWCACHE_H
