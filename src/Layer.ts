export type Input = number[];
export type Output = number[];
export type InputError = number[];
export type OutputError = number[];
export type LearningRate = number;

export abstract class Layer {
  constructor(inputSize: number, outputSize: number) {
    if (inputSize <= 0) {
      throw new Error("inputSize must be greater than 0");
    }
    if (outputSize <= 0) {
      throw new Error("outputSize must be greater than 0");
    }
  }

  abstract forwardPropagation(input: Input): Output;
  abstract backwardPropagation(
    outputError: OutputError,
    learningRate: LearningRate
  ): InputError;
}
