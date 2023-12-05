import { Matrix, add, matrix, multiply, random, zeros } from "mathjs";
import { Layer } from "./Layer";

export class FCLayer implements Layer {
  private weights: Matrix;
  private biases: Matrix;

  constructor(n_inputs: number, n_neurons: number) {
    this.weights = random(matrix([n_inputs, n_neurons]), -1, 1);
    this.biases = matrix(zeros(n_neurons));
  }

  public forward(inputs: Matrix) {
    return add(multiply(inputs, this.weights), this.biases);
  }

  public peek() {
    return {
      weights: this.weights,
      biases: this.biases,
    };
  }
}
