type Matrix = MatrixArray;
interface MatrixArray extends Array<Matrix | number> {}

type Vector = number[];
type Scalar = number;

/**
 * Initialize n x n matrix with value
 */
export function matrix(
  rows: number,
  columns: number,
  value: number = 0
): Matrix {
  return new Array(rows).fill(0).map(() => new Array(columns).fill(value));
}

/**
 * Initialize n dimensional matrix with value
 */
export function matrixN(dim: number[], value: number = 0): Matrix {
  function helper(dimensions: number[]): Matrix {
    const updatedDimensions = dimensions.slice(1);
    if (updatedDimensions.length === 0) {
      return new Array(dimensions[0]).fill(value);
    } else {
      return new Array(dimensions[0]).fill(helper(updatedDimensions));
    }
  }

  return helper(dim);
}

/**
 * Generates random n x n matrix
 */
export function randomMatrix(rows: number, columns: number): Matrix {
  return new Array(rows)
    .fill(0)
    .map(() => new Array(columns).fill(Math.random()));
}

/**
 * Get dimensions of matrix
 */
export function dim(a: Matrix): number[] {
  function helper(matrix: Matrix, dimensions: number[]): number[] {
    const updatedDimensions = dimensions.concat(matrix.length);
    const isArray = Array.isArray(matrix[0]);
    if (!isArray) {
      return updatedDimensions;
    } else {
      return helper(matrix[0] as MatrixArray, updatedDimensions);
    }
  }

  return helper(a, []);
}

/**
 * Calculate dot product of two matrices
 */
export function dot(a: Matrix, b: Matrix | Vector): Matrix | Vector {
  const [aRows, aColumns] = dim(a);
  const [bRows, bColumns] = dim(b);

  if (aColumns !== bRows) {
    throw new Error(
      `Cannot calculate dot product of matrices with dimensions ${aRows}x${aColumns} and ${bRows}x${bColumns}`
    );
  }

  /**
   * If b is a vector, calculate dot product of a and b
   */
  if (bColumns === undefined) {
    const result: Vector = new Array(aRows).fill(0);
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < aColumns; j++) {
        result[i] += (a[i] as number[])[j] * (b[j] as number);
      }
    }
    return result;
  }

  const result: Matrix = matrix(aRows, bColumns);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bColumns; j++) {
      let sum = 0;
      for (let k = 0; k < aColumns; k++) {
        sum += (a[i] as number[])[k] * (b[k] as number[])[j];
      }
      (result[i] as number[])[j] = sum;
    }
  }

  return result;
}

/**
 * Add two matrices
 */
export function add(a: Matrix | Vector, b: Matrix | Vector): Matrix {
  const [aRows, aColumns] = dim(a);
  const [bRows, bColumns] = dim(b);

  /**
   * If a and b are vectors, add them
   */
  if (aColumns === undefined && bColumns === undefined) {
    if (aRows !== bRows) {
      throw new Error(
        `Cannot add vectors with dimensions ${aRows}x${aColumns} and ${bRows}x${bColumns}`
      );
    }

    const result: Vector = new Array(aRows).fill(0);
    for (let i = 0; i < aRows; i++) {
      result[i] = (a[i] as number) + (b[i] as number);
    }
    return result;
  }

  /**
   * If b is a vector, add b to each row of a
   */
  if (bColumns === undefined) {
    const result: Matrix = matrix(aRows, aColumns);
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < aColumns; j++) {
        (result[i] as number[])[j] = (a[i] as number[])[j] + (b[j] as number);
      }
    }
    return result;
  }

  if (aRows !== bRows || aColumns !== bColumns) {
    throw new Error(
      `Cannot add matrices with dimensions ${aRows}x${aColumns} and ${bRows}x${bColumns}`
    );
  }

  const result: Matrix = matrix(aRows, aColumns);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < aColumns; j++) {
      (result[i] as number[])[j] =
        (a[i] as number[])[j] + (b[i] as number[])[j];
    }
  }

  return result;
}

/**
 * Subtract two matrices
 */
export function subtract(a: Matrix, b: Matrix): Matrix {
  const [aRows, aColumns] = dim(a);
  const [bRows, bColumns] = dim(b);

  if (aRows !== bRows || aColumns !== bColumns) {
    throw new Error(
      `Cannot subtract matrices with dimensions ${aRows}x${aColumns} and ${bRows}x${bColumns}`
    );
  }

  const result: Matrix = matrix(aRows, aColumns);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < aColumns; j++) {
      (result[i] as number[])[j] =
        (a[i] as number[])[j] - (b[i] as number[])[j];
    }
  }
  return result;
}

/**
 * Multiply two matrices
 */
export function multiply(a: Matrix, b: Matrix): Matrix {
  const [aRows, aColumns] = dim(a);
  const [bRows, bColumns] = dim(b);

  if (aColumns !== bRows) {
    throw new Error(
      `Cannot multiply matrices with dimensions ${aRows}x${aColumns} and ${bRows}x${bColumns}`
    );
  }

  const result: Matrix = matrix(aRows, bColumns);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bColumns; j++) {
      let sum = 0;
      for (let k = 0; k < aColumns; k++) {
        sum += (a[i] as number[])[k] * (b[k] as number[])[j];
      }
      (result[i] as number[])[j] = sum;
    }
  }

  return result;
}

/**
 * Transpose matrix
 */
export function transpose(a: Matrix): Matrix {
  const [rows, columns] = dim(a);
  const result: Matrix = matrix(columns, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      (result[j] as number[])[i] = (a[i] as number[])[j];
    }
  }
  return result;
}

export const roundValues = (a: Matrix, precision: number): Matrix => {
  const [rows, columns] = dim(a);
  const result: Matrix = matrix(rows, columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      (result[i] as number[])[j] =
        Math.round(((a[i] as number[])[j] + Number.EPSILON) * 10 ** precision) /
        10 ** precision;
    }
  }
  return result;
};
