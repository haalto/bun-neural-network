import { describe, expect, it } from "bun:test";
import {
  add,
  dim,
  dot,
  matrix,
  matrixN,
  multiply,
  randomMatrix,
  roundValues,
  subtract,
  transpose,
} from "./matrix";

describe("matrix", () => {
  it("should create matrix", () => {
    const result = randomMatrix(3, 3);
    expect(result.length).toEqual(3);
    expect((result[0] as number[]).length).toEqual(3);
  });

  it("should create 3x4 matrix", () => {
    const result = matrix(3, 4);
    expect(result.length).toEqual(3);
    expect((result[0] as number[]).length).toEqual(4);
  });

  it("should create 4x3x5 matrix", () => {
    const result = matrixN([4, 3, 5]);
    expect(dim(result)).toEqual([4, 3, 5]);
  });
});

describe("dim", () => {
  it("should return dimensions of matrix", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];

    const result = dim(a);
    expect(result).toEqual([3, 3]);
  });

  it("should return dimensions of matrix", () => {
    const a = [
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [
        [7, 8, 9],
        [10, 11, 12],
      ],
    ];

    const result = dim(a);
    expect(result).toEqual([2, 2, 3]);
  });
});

describe("dot", () => {
  it("should calculate dot product of two matrices", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];

    const result = dot(a, b);
    expect(result).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("should throw error if dimensions are not compatible", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8],
      [9, 10],
    ];

    expect(() => dot(a, b)).toThrow(
      "Cannot calculate dot product of matrices with dimensions 2x3 and 2x2"
    );
  });

  it("should calculate dot product for 3x3 matrices", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const b = [
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
    ];

    const result = dot(a, b);
    expect(result).toEqual([
      [66, 72, 78],
      [156, 171, 186],
      [246, 270, 294],
    ]);
  });
});

describe("add", () => {
  it("should add two matrices", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8, 9],
      [10, 11, 12],
    ];

    const result = add(a, b);
    expect(result).toEqual([
      [8, 10, 12],
      [14, 16, 18],
    ]);
  });
});

describe("subtract", () => {
  it("should subtract two matrices", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8, 9],
      [10, 11, 12],
    ];

    const result = subtract(a, b);
    expect(result).toEqual([
      [-6, -6, -6],
      [-6, -6, -6],
    ]);
  });
});

describe("multiply", () => {
  it("should multiply two matrices", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];

    const result = multiply(a, b);
    expect(result).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });
});

describe("transpose", () => {
  it("should transpose matrix", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];

    const result = transpose(a);
    expect(result).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("combine operations on matrices", () => {
  it("should count dot product of two matrices and add third matrix", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    const c = [
      [13, 14],
      [15, 16],
    ];

    const dotProduct = dot(a, b);
    expect(dotProduct).toEqual([
      [58, 64],
      [139, 154],
    ]);

    const result = add(dotProduct, c);
    expect(result).toEqual([
      [71, 78],
      [154, 170],
    ]);
  });

  it("should multiply two matrices with same dimensions", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];

    const b = [
      [7, 8, 9],
      [10, 11, 12],
    ];

    const bT = transpose(b);
    expect(bT).toEqual([
      [7, 10],
      [8, 11],
      [9, 12],
    ]);

    const result = multiply(a, bT);
    expect(result).toEqual([
      [50, 68],
      [122, 167],
    ]);
  });

  it("should calculate dot product for two vectors", () => {
    const a = [[1, 2, 3]];
    const b = [[4, 5, 6]];

    const result = dot(a, transpose(b));
    expect(result).toEqual([[32]]);
  });

  it("should calculate dot product of two matrices and transpose result with decimals and add vector", () => {
    const a = [1.0, 2.0, 3.0, 2.5];
    const b = [
      [0.2, 0.8, -0.5, 1.0],
      [0.5, -0.91, 0.26, -0.5],
      [-0.26, -0.27, 0.17, 0.87],
    ];

    const c = [2.0, 3.0, 0.5];

    const dotProduct = dot(b, a);
    const result = add(dotProduct, c);
    expect(result).toEqual([4.8, 1.21, 2.385]);
  });

  it("should calculate dot product for two 3x4 matrices and add vector", () => {
    const a = [
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8],
    ];

    const b = [
      [0.2, 0.8, -0.5, 1.0],
      [0.5, -0.91, 0.26, -0.5],
      [-0.26, -0.27, 0.17, 0.87],
    ];

    const c = [2.0, 3.0, 0.5];

    const dotProduct = dot(a, transpose(b));
    const result = add(dotProduct, c);

    const rounded = roundValues(result, 3);
    expect(rounded).toEqual([
      [4.8, 1.21, 2.385],
      [8.9, -1.81, 0.2],
      [1.41, 1.051, 0.026],
    ]);
  });
});
