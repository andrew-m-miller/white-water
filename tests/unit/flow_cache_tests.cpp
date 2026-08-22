// Focused Phase 2.2 policy tests.  This file intentionally uses no test framework so the cache
// can be exercised in the same dependency-free build as the core algebra.

#include "core/flow/FlowCache.h"

#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {

using whitewater::AccumulatedFieldValue;
using whitewater::FieldGeometry;
using whitewater::FlowCache;
using whitewater::FlowCacheBudgets;
using whitewater::FlowCacheKey;
using whitewater::FlowField;
using whitewater::FlowLink;
using whitewater::ScalarField;
using whitewater::Vec2;

int failures = 0;

void require(bool condition, const char *message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

FlowField fieldOf(int columns, int rows, const FieldGeometry &geometry, float value = 0.0f) {
  FlowField field(columns, rows, geometry);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float *node = field.node(column, row);
      node[0] = value;
      node[1] = value * 2.0f;
    }
  }
  return field;
}

std::shared_ptr<const ScalarField> confidenceOf(int columns, int rows,
                                                const FieldGeometry &geometry) {
  auto confidence = std::make_shared<ScalarField>(columns, rows, geometry);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      confidence->node(column, row)[0] = 1.0f;
    }
  }
  return confidence;
}

std::shared_ptr<const FlowLink> linkOf(int from, int to, int columns, int rows,
                                       const FieldGeometry &geometry,
                                       const std::string &fingerprint, float value = 0.0f) {
  return std::make_shared<FlowLink>(from, to, geometry, geometry,
                                    fieldOf(columns, rows, geometry, value), fingerprint);
}

FlowCacheKey keyOf(FlowCache::Generation generation, int from, int to, int columns, int rows,
                   const FieldGeometry &geometry, const std::string &model = "model-a") {
  FlowCacheKey key;
  key.fromFrame = from;
  key.toFrame = to;
  key.generation = generation;
  key.modelFingerprint = model;
  key.matteToken = "matte-none";
  key.inputConditioningToken = "conditioning-v1";
  key.modelParametersFingerprint = "params-v1";
  key.modelColumns = columns;
  key.modelRows = rows;
  key.modelGeometry = geometry;
  key.destinationGeometry = geometry;
  return key;
}

AccumulatedFieldValue accumulatedOf(int from, int to, int columns, int rows,
                                    const FieldGeometry &geometry,
                                    bool withConfidence = false) {
  auto link = linkOf(from, to, columns, rows, geometry, "model-a", 1.0f);
  if (!withConfidence) return AccumulatedFieldValue(link);
  return AccumulatedFieldValue(link, confidenceOf(columns, rows, geometry));
}

void testExactAccountingAndReplacement() {
  const FieldGeometry geometry(Vec2(0.5, 0.5), 1.0, 1.0);
  const auto token = FlowCache(1).captureGeneration();

  const auto plainLink = linkOf(0, 1, 2, 2, geometry, "model-a");
  const auto confidence = confidenceOf(2, 2, geometry);
  whitewater::PairwiseFlowValue withConfidence(plainLink, confidence);
  require(withConfidence.byteSize() == plainLink->field().byteSize() + confidence->byteSize(),
          "pairwise accounting includes optional confidence bytes");

  FlowCache cache(128, 128);
  const FlowCacheKey key = keyOf(token.value, 0, 1, 2, 2, geometry);
  require(cache.putPairwise(key, withConfidence), "pairwise value fits and is retained");
  require(cache.pairwiseSize() == 1, "pairwise replacement test starts with one entry");
  require(cache.pairwiseByteCount() == withConfidence.byteSize(),
          "pairwise byte count equals exact payload size");

  // Replacing the same key subtracts the old payload before adding the new one; it does not
  // double-count a confidence field or leave two LRU nodes for one identity.
  const auto replacement = linkOf(0, 1, 2, 2, geometry, "model-a", 3.0f);
  whitewater::PairwiseFlowValue replacementValue(replacement);
  require(cache.putPairwise(key, replacementValue), "same-key pairwise replacement succeeds");
  require(cache.pairwiseSize() == 1, "replacement keeps one LRU entry");
  require(cache.pairwiseByteCount() == replacementValue.byteSize(),
          "replacement byte accounting removes the prior value");
  auto hit = cache.lookupPairwise(key);
  require(hit.has_value() &&
              whitewater::flowNode(hit->link->field(), 0, 0) == Vec2(3.0, 6.0),
          "replacement returns the newest value");
}

void testRecencyAndSeparateStores() {
  const FieldGeometry geometry(Vec2(-3.5, 2.5), 2.0, 0.5);
  FlowCache cache(96, 64);
  const auto generation = cache.captureGeneration().value;

  // 2x2 FlowField = 32 bytes, 3x2 = 48 bytes.  Touching A before adding C makes B the only
  // least-recently-used victim when the 96-byte pairwise budget is exceeded.
  const FlowCacheKey keyA = keyOf(generation, 0, 1, 2, 2, geometry);
  const FlowCacheKey keyB = keyOf(generation, 1, 2, 3, 2, geometry);
  const FlowCacheKey keyC = keyOf(generation, 2, 3, 2, 2, geometry);
  require(cache.putPairwise(keyA, whitewater::PairwiseFlowValue(linkOf(
                               0, 1, 2, 2, geometry, "model-a"))),
          "A fits pairwise budget");
  require(cache.putPairwise(keyB, whitewater::PairwiseFlowValue(linkOf(
                               1, 2, 3, 2, geometry, "model-a"))),
          "B fits pairwise budget");
  require(cache.lookupPairwise(keyA).has_value(), "lookup promotes A to most recent");
  require(cache.putPairwise(keyC, whitewater::PairwiseFlowValue(linkOf(
                               2, 3, 2, 2, geometry, "model-a"))),
          "C fits after LRU eviction");
  require(cache.containsPairwise(keyA), "recent A survives eviction");
  require(!cache.containsPairwise(keyB), "least-recent B is evicted");
  require(cache.containsPairwise(keyC), "newest C is retained");
  require(cache.pairwiseByteCount() == 64, "pairwise eviction leaves exact 64-byte total");

  const FlowCacheKey accumulatedKey = keyOf(generation, 7, 9, 2, 2, geometry);
  const AccumulatedFieldValue accumulated = accumulatedOf(7, 9, 2, 2, geometry, true);
  require(cache.putAccumulated(accumulatedKey, accumulated),
          "accumulated value uses its distinct store");
  require(cache.accumulatedByteCount() == accumulated.byteSize(),
          "accumulated bytes are tracked independently");
  require(cache.pairwiseByteCount() == 64,
          "accumulated insertion does not perturb pairwise LRU accounting");
}

void testKeyIdentity() {
  const FieldGeometry geometry(Vec2(0.5, 0.5), 1.0, 1.0);
  FlowCache cache(1024);
  const auto generation = cache.captureGeneration().value;
  const FlowCacheKey base = keyOf(generation, 0, 1, 2, 2, geometry);
  require(cache.putPairwise(base, whitewater::PairwiseFlowValue(
                               linkOf(0, 1, 2, 2, geometry, "model-a"))),
          "base key is retained");

  FlowCacheKey changed = base;
  changed.toFrame = 2;
  require(!cache.containsPairwise(changed), "endpoints participate in cache identity");
  changed = base;
  changed.modelFingerprint = "model-b";
  require(!cache.containsPairwise(changed), "model fingerprint participates in identity");
  changed = base;
  changed.matteToken = "matte-white";
  require(!cache.containsPairwise(changed), "matte token participates in identity");
  changed = base;
  changed.inputConditioningToken = "conditioning-v2";
  require(!cache.containsPairwise(changed), "input conditioning participates in identity");
  changed = base;
  changed.modelParametersFingerprint = "params-v2";
  require(!cache.containsPairwise(changed), "model parameter fingerprint participates in identity");
  changed = base;
  changed.modelGeometry.spacingX = 2.0;
  require(!cache.containsPairwise(changed), "source model geometry participates in identity");
  changed = base;
  changed.destinationGeometry.spacingY = 2.0;
  require(!cache.containsPairwise(changed),
          "destination model geometry participates in identity");
  changed = base;
  changed.schemaVersion += 1;
  require(!cache.containsPairwise(changed), "cache schema participates in identity");
}

void testDisabledAndOversize() {
  const FieldGeometry geometry(Vec2(0.5, 0.5), 1.0, 1.0);
  const auto generation = FlowCache().captureGeneration().value;
  const FlowCacheKey key = keyOf(generation, 0, 1, 2, 2, geometry);
  const whitewater::PairwiseFlowValue value(linkOf(0, 1, 2, 2, geometry, "model-a"));

  FlowCache disabled(0, 0);
  require(!disabled.putPairwise(key, value), "zero budget disables pairwise cache");
  require(disabled.pairwiseSize() == 0 && disabled.pairwiseByteCount() == 0,
          "disabled cache retains no pairwise entry");
  require(!disabled.putAccumulated(key, accumulatedOf(0, 1, 2, 2, geometry)),
          "zero budget disables accumulated cache");

  // The pairwise payload is 32 bytes and the accumulated payload is also 32 bytes.  Neither
  // oversize single entry is admitted; unlike a soft target, the configured budget is exact.
  FlowCache oversize(31, 31);
  require(!oversize.putPairwise(key, value), "oversize pairwise entry is rejected");
  require(!oversize.putAccumulated(key, accumulatedOf(0, 1, 2, 2, geometry)),
          "oversize accumulated entry is rejected");
  require(oversize.totalByteCount() == 0, "oversize entries consume no budget");
}

void testKeyValueConsistency() {
  const FieldGeometry geometry(Vec2(0.5, 0.5), 1.0, 1.0);
  FlowCache cache(1024);
  const auto token = cache.captureGeneration();
  const FlowCacheKey key = keyOf(token.value, 0, 1, 2, 2, geometry);
  const whitewater::PairwiseFlowValue correct(
      linkOf(0, 1, 2, 2, geometry, "model-a"));
  require(cache.publishPairwise(token, key, correct), "matching key/value publishes");

  FlowCacheKey wrongEndpoint = key;
  wrongEndpoint.toFrame = 2;
  require(!cache.publishPairwise(token, wrongEndpoint, correct),
          "pairwise endpoint mismatch is rejected");
  FlowCacheKey wrongModel = key;
  wrongModel.modelFingerprint = "model-b";
  require(!cache.publishPairwise(token, wrongModel, correct),
          "pairwise model mismatch is rejected");
  FlowCacheKey wrongShape = key;
  wrongShape.modelColumns = 3;
  require(!cache.publishPairwise(token, wrongShape, correct),
          "pairwise lattice shape mismatch is rejected");
  FlowCacheKey wrongGeometry = key;
  wrongGeometry.modelGeometry.spacingX = 2.0;
  require(!cache.publishPairwise(token, wrongGeometry, correct),
          "pairwise lattice geometry mismatch is rejected");
  FlowCacheKey wrongDestinationGeometry = key;
  wrongDestinationGeometry.destinationGeometry.spacingY = 2.0;
  require(!cache.publishPairwise(token, wrongDestinationGeometry, correct),
          "pairwise destination geometry mismatch is rejected");

  const auto mismatchedConfidence = confidenceOf(3, 2, geometry);
  require(!cache.publishPairwise(
              token, key,
              whitewater::PairwiseFlowValue(linkOf(0, 1, 2, 2, geometry, "model-a"),
                                            mismatchedConfidence)),
          "pairwise confidence shape mismatch is rejected");
  require(!cache.publishAccumulated(
              token, key,
              AccumulatedFieldValue(linkOf(0, 1, 3, 2, geometry, "model-a"))),
          "accumulated field shape mismatch is rejected");
  require(!cache.publishAccumulated(
              token, key,
              AccumulatedFieldValue(linkOf(0, 1, 2, 2, geometry, "model-a"),
                                    mismatchedConfidence)),
          "accumulated confidence shape mismatch is rejected");
}

void testGenerationAndStalePublication() {
  const FieldGeometry geometry(Vec2(0.5, 0.5), 1.0, 1.0);
  FlowCache cache(1024);
  const FlowCache::GenerationToken oldToken = cache.captureGeneration();
  const FlowCacheKey oldKey = keyOf(oldToken.value, 0, 1, 2, 2, geometry);
  const whitewater::PairwiseFlowValue value(linkOf(0, 1, 2, 2, geometry, "model-a"));
  require(cache.publishPairwise(oldToken, oldKey, value), "initial generation publishes");
  require(cache.containsPairwise(oldKey), "initial generation entry is visible");

  const FlowCache::Generation newGeneration = cache.bumpGeneration();
  require(newGeneration != oldToken.value, "generation bump advances token");
  require(!cache.containsPairwise(oldKey), "generation bump invalidates old entries");
  require(cache.pairwiseByteCount() == 0 && cache.accumulatedByteCount() == 0,
          "generation bump clears both stores");

  std::mutex gateMutex;
  std::condition_variable gate;
  bool workerReady = false;
  bool releaseWorker = false;
  bool stalePublished = true;
  std::thread worker([&] {
    {
      std::lock_guard<std::mutex> lock(gateMutex);
      workerReady = true;
    }
    gate.notify_one();
    std::unique_lock<std::mutex> lock(gateMutex);
    gate.wait(lock, [&] { return releaseWorker; });
    lock.unlock();
    stalePublished = cache.publishPairwise(oldToken, oldKey, value);
  });

  {
    std::unique_lock<std::mutex> lock(gateMutex);
    gate.wait(lock, [&] { return workerReady; });
  }
  // The worker has finished its synthetic out-of-lock computation but has not published yet.
  // This ordering makes the stale-result race deterministic rather than a sleep-based test.
  cache.bumpGeneration();
  {
    std::lock_guard<std::mutex> lock(gateMutex);
    releaseWorker = true;
  }
  gate.notify_one();
  worker.join();
  require(!stalePublished, "old in-flight generation cannot publish after bump");

  const FlowCache::GenerationToken currentToken = cache.captureGeneration();
  const FlowCacheKey currentKey = keyOf(currentToken.value, 0, 1, 2, 2, geometry);
  require(cache.publishPairwise(currentToken, currentKey, value),
          "current generation publishes after stale result loses race");
  require(cache.containsPairwise(currentKey), "current generation result is resident");

  cache.clear();
  require(cache.generation() != currentToken.value, "clear advances the generation token");
  require(cache.pairwiseSize() == 0, "clear drops resident pairwise values");
  require(!cache.publishPairwise(currentToken, currentKey, value),
          "work captured before Clear cannot repopulate the cache");
}

}  // namespace

int main() {
  testExactAccountingAndReplacement();
  testRecencyAndSeparateStores();
  testKeyIdentity();
  testDisabledAndOversize();
  testKeyValueConsistency();
  testGenerationAndStalePublication();

  if (failures != 0) {
    std::cerr << failures << " flow cache test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "flow cache tests passed\n";
  return EXIT_SUCCESS;
}
