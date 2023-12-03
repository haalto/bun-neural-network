import { Input, InputError, Layer, Output, OutputError } from "./Layer";

export class FCLayer implements Layer {
  constructor(protected inputSize: number, protected outputSize: number) {
    if (inputSize <= 0) {
      throw new Error("inputSize must be greater than 0");
    }
    if (outputSize <= 0) {
      throw new Error("outputSize must be greater than 0");
    }
  }

  forwardPropagation(input: Input): Output {
    return input as Output;
  }

  backwardPropagation(
    outputError: OutputError,
    learningRate: number
  ): OutputError {
    return outputError as InputError;
  }
}
