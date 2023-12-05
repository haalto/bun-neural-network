import { Matrix } from "mathjs";
import { relu } from "../../math/activation";
import { Layer } from "../Layer";

export class ReLU implements Layer {
  forward(input: Matrix) {
    return relu(input);
  }
}
