import { Matrix, exp, sum, map, dotDivide } from "mathjs";

export function relu(input: Matrix) {
  return input.map((value) => Math.max(0, value));
}

/**
 * Computes the softmax of a matrix per row.
 */
export function softmax(logits: Matrix, axis = 1) {
  const expValues = map(logits, (l) => exp(l));
  const denom = sum(expValues, axis) as unknown as Matrix; // Type inference bug it returns one dimensional vector
  const denomResized = denom.resize([denom.size()[0], 1]); // Resize to N x 1 matrix to be able to divide
  return dotDivide(expValues, denomResized);

  /* //Iterative implementation
  const probabilities = expValues.map((ev, i) => {
    const denomValue = denom.get([i[0]]);
    return ev / denomValue;
  });

  return probabilities;
  */
}
