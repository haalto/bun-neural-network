import { describe, expect, it } from "bun:test";
import { add, matrix } from "mathjs";
import { relu, softmax } from "./activation";

describe("ReLU activation function", () => {
  it("should return a matrix with the same shape as the input matrix", () => {
    const a = matrix([
      [1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8],
    ]);

    const result = relu(a);

    expect(result.size()).toEqual(a.size());
  });

  it("should return a matrix with the same values as the input matrix where the values are greater than 0", () => {
    const a = matrix([
      [1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8],
    ]);

    const result = relu(a);

    expect(result.toArray()).toEqual([
      [1, 2, 3, 2.5],
      [2.0, 5.0, 0, 2.0],
      [0, 2.7, 3.3, 0],
    ]);
  });
});

describe.only("Softmax activation function", () => {
  it("should return a matrix with the same shape as the input matrix", () => {
    const a = matrix([
      [1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8],
    ]);

    const result = softmax(a);
    expect(result.size()).toEqual(a.size());
  });

  it("probabilities should sum to 1", () => {
    const inputMatrix = matrix([
      [2, 1, 0.1],
      [1, 2, 0.1],
      [0.1, 1, 2],
    ]);

    const result = softmax(inputMatrix);

    const expectedOutput = matrix([
      [0.6590011388859679, 0.2424329707047139, 0.09856589040931818],
      [0.2424329707047139, 0.6590011388859679, 0.09856589040931818],
      [0.09856589040931818, 0.2424329707047139, 0.6590011388859679],
    ]);

    expect(result.toArray()).toEqual(expectedOutput.toArray());
  });
});
