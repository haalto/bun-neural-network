import { Matrix } from "mathjs";

export abstract class Layer {
  abstract forward(input: Matrix): Matrix;
}
