import { describe, expect, it } from "bun:test";
import {
  matrix,
  dot,
  add,
  transpose,
  multiply,
  dotMultiply,
  sum,
  round,
} from "mathjs";

/**
 * These tests are for playing around with mathjs and comparing it to numpy operations
 * following sentdex's tutorial for building Neural Networks from scratch
 */
describe("test mathjs for matrix calculation and compare it to numpy", () => {
  it("should be able to calculate dot product and add bias", () => {
    const inputs = [1, 2, 3];
    const weights = [4, 5, 6];
    const bias = 2;
    const dotProduct = dot(inputs, weights) + bias;
    expect(dotProduct).toBe(34);
  });

  it("should be able to calculate dot product and add bias", () => {
    const inputs = [1.0, 2.0, 3.0, 2.5];
    const weights = matrix([0.2, 0.8, -0.5, 1.0]);
    const bias = 2;
    const dotProduct = dot(inputs, weights) + bias;
    expect(dotProduct).toBe(4.8);
  });

  it("should be able to calculate dot product and add bias", () => {
    const inputs = [1.0, 2.0, 3.0, 2.5];
    const weights = matrix([
      [0.2, 0.8, -0.5, 1.0],
      [0.5, -0.91, 0.26, -0.5],
      [-0.26, -0.27, 0.17, 0.87],
    ]);
    const bias = [2, 3, 0.5];

    // In numpy, this would be: np.dot(inputs, weights) + bias. Numpy uses sum product if a is N-D and b is 1-D
    const outputs = add(sum(dotMultiply(weights, inputs), 1), bias);
    expect(outputs).toStrictEqual(matrix([4.8, 1.21, 2.385]));
  });

  it("should be able to calculate dot product for two 3x4 matrices and add vector", () => {
    const inputs = matrix([
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8],
    ]);

    const weights = matrix([
      [0.2, 0.8, -0.5, 1.0],
      [0.5, -0.91, 0.26, -0.5],
      [-0.26, -0.27, 0.17, 0.87],
    ]);

    const bias = [2.0, 3.0, 0.5];

    // In numpy, this would be: np.dot(inputs, weights) + bias. Numpy uses matrix multiplication if both a and b are N-D
    const dotProduct = multiply(inputs, transpose(weights)) as math.Matrix;
    const result = add(dotProduct, matrix(bias));
    expect(round(result, 3)).toStrictEqual(
      matrix([
        [4.8, 1.21, 2.385],
        [8.9, -1.81, 0.2],
        [1.41, 1.051, 0.026],
      ])
    );
  });
});
