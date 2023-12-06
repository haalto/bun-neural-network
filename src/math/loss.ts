import {
  Matrix,
  deepEqual,
  divide,
  dotMultiply,
  log,
  map,
  max,
  sum,
} from "mathjs";
import { argmax } from "./helpers";

/**
 * Calculates the categorical cross entropy loss between the predictions and the labels.
 */
export function categoricalCrossEntropy(predictions: Matrix, labels: Matrix) {
  const epsilon = 1e-15; // Small constant to avoid log(0)

  // Ensure predictions and labels are matrices of the same size
  if (!deepEqual(predictions.size(), labels.size())) {
    throw new Error("Input matrices must have the same size.");
  }

  // Clip predictions to avoid log(0)
  const clippedPredictions = map(predictions, (value) => max(value, epsilon));

  // Calculate the negative log likelihood
  const logLikelihoods = dotMultiply(
    labels,
    clippedPredictions.map((v) => log(v))
  );

  const batchLoss = divide(sum(logLikelihoods), -predictions.size()[0]);

  return batchLoss;
}

/**
 * Calculates the accuracy of the predictions.
 */
export function accuracyHotOneEncoded(
  predictions: Matrix,
  labels: Matrix
): number {
  const predArgmax = argmax(predictions.toArray() as number[][]);
  const labelArgmax = argmax(labels.toArray() as number[][]);

  if (predArgmax === null || labelArgmax === null) {
    throw new Error("Empty matrix");
  }

  let correct = 0;
  for (let i = 0; i < predArgmax.length; i++) {
    if (predArgmax[i] === labelArgmax[i]) {
      correct++;
    }
  }

  return correct / predArgmax.length;
}
