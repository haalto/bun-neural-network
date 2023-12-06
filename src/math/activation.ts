import { Matrix, exp, sum, map, dotDivide } from "mathjs";

/**
 * Computes the rectified linear unit of a matrix. E.q [[-1,2],[3,-4]] -> [[0,2],[3,0]]
 */
export function relu(input: Matrix) {
  return input.map((value) => Math.max(0, value));
}

/**
 * Computes the softmax of a matrix per row. E.q [[1,2],[3,4]] -> [[0.26894142, 0.73105858],[0.26894142, 0.73105858
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
