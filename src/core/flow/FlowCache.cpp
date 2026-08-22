#include "core/flow/FlowCache.h"

#include <functional>
#include <limits>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace whitewater {
namespace {

std::size_t saturatedAdd(std::size_t a, std::size_t b) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  return a > maximum - b ? maximum : a + b;
}

bool confidenceMatches(const FlowField &field,
                       const std::shared_ptr<const ScalarField> &confidence) {
  return confidence == nullptr ||
         (!confidence->isEmpty() && confidence->columns() == field.columns() &&
          confidence->rows() == field.rows() && confidence->geometry() == field.geometry());
}

bool pairwiseValueMatchesKey(const FlowCacheKey &key, const PairwiseFlowValue &value) {
  if (!value.isValid() || key.modelColumns <= 0 || key.modelRows <= 0 ||
      !key.modelGeometry.isValid() || !key.destinationGeometry.isValid()) {
    return false;
  }
  const FlowLink &link = *value.link;
  const FlowField &field = link.field();
  return key.fromFrame == link.fromTime() && key.toFrame == link.toTime() &&
         key.modelFingerprint == link.modelFingerprint() &&
         key.modelColumns == field.columns() && key.modelRows == field.rows() &&
         key.modelGeometry == field.geometry() &&
         key.destinationGeometry == link.toGeometry() &&
         confidenceMatches(field, value.confidence);
}

bool accumulatedValueMatchesKey(const FlowCacheKey &key,
                                const AccumulatedFieldValue &value) {
  if (!value.isValid() || key.modelColumns <= 0 || key.modelRows <= 0 ||
      !key.modelGeometry.isValid() || !key.destinationGeometry.isValid()) {
    return false;
  }
  const FlowLink &link = *value.link;
  const FlowField &field = link.field();
  return key.fromFrame == link.fromTime() && key.toFrame == link.toTime() &&
         key.modelFingerprint == link.modelFingerprint() &&
         key.modelColumns == field.columns() && key.modelRows == field.rows() &&
         key.modelGeometry == field.geometry() &&
         key.destinationGeometry == link.toGeometry() &&
         confidenceMatches(field, value.confidence);
}

template <typename T>
void hashCombine(std::size_t &seed, const T &value) {
  // This is intentionally a local hash-combine policy.  FlowCacheKey never leaves the process,
  // so a stable serialized hash is unnecessary; equality remains the correctness check.
  const std::size_t valueHash = std::hash<T>{}(value);
  seed ^= valueHash + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2);
}

template <typename Value>
class LruStore {
 public:
  using EntryValue = Value;

  explicit LruStore(std::size_t byteBudget = 0) : byteBudget_(byteBudget) {}

  std::optional<Value> get(const FlowCacheKey &key) {
    const auto found = index_.find(key);
    if (found == index_.end()) return std::nullopt;
    order_.splice(order_.begin(), order_, found->second);
    return found->second->value;
  }

  bool contains(const FlowCacheKey &key) const { return index_.find(key) != index_.end(); }

  // The caller has already done value-specific validation and byte accounting.  Keeping this
  // policy generic is what lets pairwise links and accumulated fields remain separate stores
  // without duplicating the eviction rules.
  bool put(const FlowCacheKey &key, Value value, std::size_t bytes) {
    erase(key);
    if (byteBudget_ == 0 || bytes == 0 || bytes > byteBudget_) return false;

    order_.push_front(Entry{key, std::move(value), bytes});
    index_.emplace(order_.front().key, order_.begin());
    bytes_ = saturatedAdd(bytes_, bytes);
    evictToFit();
    return index_.find(key) != index_.end();
  }

  void setBudget(std::size_t byteBudget) {
    byteBudget_ = byteBudget;
    if (byteBudget_ == 0) {
      clear();
      return;
    }
    evictToFit();
  }

  void clear() {
    order_.clear();
    index_.clear();
    bytes_ = 0;
  }

  std::size_t byteCount() const { return bytes_; }
  std::size_t size() const { return order_.size(); }
  std::size_t byteBudget() const { return byteBudget_; }

 private:
  struct Entry {
    FlowCacheKey key;
    Value value;
    std::size_t bytes = 0;
  };

  void erase(const FlowCacheKey &key) {
    const auto found = index_.find(key);
    if (found == index_.end()) return;
    bytes_ -= found->second->bytes;
    order_.erase(found->second);
    index_.erase(found);
  }

  void evictToFit() {
    while (bytes_ > byteBudget_ && !order_.empty()) {
      auto victim = std::prev(order_.end());
      bytes_ -= victim->bytes;
      index_.erase(victim->key);
      order_.erase(victim);
    }
  }

  std::size_t byteBudget_ = 0;
  std::size_t bytes_ = 0;
  // Front is most-recently-used; back is the next eviction victim.  List iterators remain valid
  // when another entry is touched by splice, so map lookup and recency updates are both O(1).
  std::list<Entry> order_;
  std::unordered_map<FlowCacheKey, typename std::list<Entry>::iterator, FlowCacheKeyHash> index_;
};

}  // namespace

bool FlowCacheKey::operator==(const FlowCacheKey &other) const {
  return fromFrame == other.fromFrame && toFrame == other.toFrame && generation == other.generation &&
         modelFingerprint == other.modelFingerprint && matteToken == other.matteToken &&
         inputConditioningToken == other.inputConditioningToken &&
         modelParametersFingerprint == other.modelParametersFingerprint &&
         modelColumns == other.modelColumns && modelRows == other.modelRows &&
         modelGeometry == other.modelGeometry &&
         destinationGeometry == other.destinationGeometry && schemaVersion == other.schemaVersion;
}

std::size_t FlowCacheKeyHash::operator()(const FlowCacheKey &key) const noexcept {
  std::size_t seed = 0;
  hashCombine(seed, key.fromFrame);
  hashCombine(seed, key.toFrame);
  hashCombine(seed, key.generation);
  hashCombine(seed, key.modelFingerprint);
  hashCombine(seed, key.matteToken);
  hashCombine(seed, key.inputConditioningToken);
  hashCombine(seed, key.modelParametersFingerprint);
  hashCombine(seed, key.modelColumns);
  hashCombine(seed, key.modelRows);
  hashCombine(seed, key.modelGeometry.origin.x);
  hashCombine(seed, key.modelGeometry.origin.y);
  hashCombine(seed, key.modelGeometry.spacingX);
  hashCombine(seed, key.modelGeometry.spacingY);
  hashCombine(seed, key.destinationGeometry.origin.x);
  hashCombine(seed, key.destinationGeometry.origin.y);
  hashCombine(seed, key.destinationGeometry.spacingX);
  hashCombine(seed, key.destinationGeometry.spacingY);
  hashCombine(seed, key.schemaVersion);
  return seed;
}

PairwiseFlowValue::PairwiseFlowValue(const FlowLink &linkValue,
                                     std::shared_ptr<const ScalarField> confidenceValue)
    : link(std::make_shared<FlowLink>(linkValue)), confidence(std::move(confidenceValue)) {}

std::size_t PairwiseFlowValue::byteSize() const {
  if (!isValid()) return 0;
  return saturatedAdd(link->field().byteSize(), confidence ? confidence->byteSize() : 0);
}

AccumulatedFieldValue::AccumulatedFieldValue(const FlowLink &linkValue,
                                             std::shared_ptr<const ScalarField> confidenceValue)
    : link(std::make_shared<FlowLink>(linkValue)), confidence(std::move(confidenceValue)) {}

std::size_t AccumulatedFieldValue::byteSize() const {
  if (!isValid()) return 0;
  return saturatedAdd(link->field().byteSize(), confidence ? confidence->byteSize() : 0);
}

struct FlowCache::State {
  mutable std::mutex mutex;
  Generation generation = 0;
  LruStore<PairwiseFlowValue> pairwise;
  LruStore<AccumulatedFieldValue> accumulated;

  explicit State(FlowCacheBudgets budgets)
      : pairwise(budgets.pairwiseBytes), accumulated(budgets.accumulatedBytes) {}
};

FlowCache::FlowCache() : state_(std::make_unique<State>(FlowCacheBudgets{})) {}

FlowCache::FlowCache(FlowCacheBudgets budgets) : state_(std::make_unique<State>(budgets)) {}

FlowCache::FlowCache(std::size_t perStoreByteBudget)
    : FlowCache(FlowCacheBudgets{perStoreByteBudget, perStoreByteBudget}) {}

FlowCache::FlowCache(std::size_t pairwiseByteBudget, std::size_t accumulatedByteBudget)
    : FlowCache(FlowCacheBudgets{pairwiseByteBudget, accumulatedByteBudget}) {}

FlowCache::~FlowCache() = default;
FlowCache::FlowCache(FlowCache &&) noexcept = default;
FlowCache &FlowCache::operator=(FlowCache &&) noexcept = default;

FlowCache::Generation FlowCache::generation() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->generation;
}

FlowCache::GenerationToken FlowCache::captureGeneration() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return GenerationToken{state_->generation};
}

FlowCache::Generation FlowCache::bumpGeneration() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  ++state_->generation;
  state_->pairwise.clear();
  state_->accumulated.clear();
  return state_->generation;
}

void FlowCache::clear() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  ++state_->generation;
  state_->pairwise.clear();
  state_->accumulated.clear();
}

void FlowCache::setBudgets(FlowCacheBudgets budgets) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->pairwise.setBudget(budgets.pairwiseBytes);
  state_->accumulated.setBudget(budgets.accumulatedBytes);
}

void FlowCache::setPairwiseByteBudget(std::size_t byteBudget) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->pairwise.setBudget(byteBudget);
}

void FlowCache::setAccumulatedByteBudget(std::size_t byteBudget) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accumulated.setBudget(byteBudget);
}

FlowCacheBudgets FlowCache::budgets() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return FlowCacheBudgets{state_->pairwise.byteBudget(), state_->accumulated.byteBudget()};
}

std::size_t FlowCache::pairwiseByteCount() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->pairwise.byteCount();
}

std::size_t FlowCache::accumulatedByteCount() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->accumulated.byteCount();
}

std::size_t FlowCache::totalByteCount() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return saturatedAdd(state_->pairwise.byteCount(), state_->accumulated.byteCount());
}

std::size_t FlowCache::pairwiseSize() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->pairwise.size();
}

std::size_t FlowCache::accumulatedSize() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->accumulated.size();
}

std::optional<PairwiseFlowValue> FlowCache::lookupPairwise(const FlowCacheKey &key) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (key.generation != state_->generation) return std::nullopt;
  return state_->pairwise.get(key);
}

std::optional<AccumulatedFieldValue> FlowCache::lookupAccumulated(const FlowCacheKey &key) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (key.generation != state_->generation) return std::nullopt;
  return state_->accumulated.get(key);
}

bool FlowCache::containsPairwise(const FlowCacheKey &key) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return key.generation == state_->generation && state_->pairwise.contains(key);
}

bool FlowCache::containsAccumulated(const FlowCacheKey &key) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return key.generation == state_->generation && state_->accumulated.contains(key);
}

bool FlowCache::putPairwise(const FlowCacheKey &key, PairwiseFlowValue value) {
  const std::size_t bytes = value.byteSize();
  const bool consistent = pairwiseValueMatchesKey(key, value);
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (key.generation != state_->generation) return false;
  if (!consistent) {
    // Calling put with an invalid value is an explicit replacement attempt.  Do not leave a
    // previous value under the key that a caller just failed to recompute.
    state_->pairwise.put(key, std::move(value), 0);
    return false;
  }
  return state_->pairwise.put(key, std::move(value), bytes);
}

bool FlowCache::putAccumulated(const FlowCacheKey &key, AccumulatedFieldValue value) {
  const std::size_t bytes = value.byteSize();
  const bool consistent = accumulatedValueMatchesKey(key, value);
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (key.generation != state_->generation) return false;
  if (!consistent) {
    state_->accumulated.put(key, std::move(value), 0);
    return false;
  }
  return state_->accumulated.put(key, std::move(value), bytes);
}

bool FlowCache::publishPairwise(GenerationToken token, const FlowCacheKey &key,
                                PairwiseFlowValue value) {
  const std::size_t bytes = value.byteSize();
  const bool consistent = pairwiseValueMatchesKey(key, value);
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (token.value != state_->generation || key.generation != token.value) return false;
  if (!consistent) {
    state_->pairwise.put(key, std::move(value), 0);
    return false;
  }
  return state_->pairwise.put(key, std::move(value), bytes);
}

bool FlowCache::publishAccumulated(GenerationToken token, const FlowCacheKey &key,
                                   AccumulatedFieldValue value) {
  const std::size_t bytes = value.byteSize();
  const bool consistent = accumulatedValueMatchesKey(key, value);
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (token.value != state_->generation || key.generation != token.value) return false;
  if (!consistent) {
    state_->accumulated.put(key, std::move(value), 0);
    return false;
  }
  return state_->accumulated.put(key, std::move(value), bytes);
}

}  // namespace whitewater
