/**
 * Get index for the maximum value in each row of a matrix.
 */
export function argmax(matrix: number[][]): number[] | null {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return null; // Empty matrix
  }

  const argmaxIndices: number[] = [];

  for (let i = 0; i < matrix.length; i++) {
    if (matrix[i].length === 0) {
      return null; // Empty row
    }

    let maxElement = matrix[i][0];
    let maxCol = 0;

    for (let j = 1; j < matrix[i].length; j++) {
      if (matrix[i][j] > maxElement) {
        maxElement = matrix[i][j];
        maxCol = j;
      }
    }

    argmaxIndices.push(maxCol);
  }

  return argmaxIndices;
}
