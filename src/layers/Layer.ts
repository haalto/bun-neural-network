import { Matrix, add, matrix, multiply, random, zeros } from "mathjs";

export class FCLayer {
  private weights: Matrix;
  private biases: Matrix;
  private output: Matrix | null = null;

  constructor(n_inputs: number, n_neurons: number) {
    this.weights = random(matrix([n_inputs, n_neurons]), -1, 1);
    this.biases = matrix(zeros(n_neurons));
  }

  public forward(inputs: Matrix) {
    this.output = add(multiply(inputs, this.weights), this.biases);
  }

  public getOutput() {
    return this.output;
  }

  public peek() {
    return {
      weights: this.weights,
      biases: this.biases,
      output: this.output,
    };
  }
}
