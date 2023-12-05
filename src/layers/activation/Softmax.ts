import { Matrix } from "mathjs";
import { softmax } from "../../math/activation";
import { Layer } from "../Layer";

export class Softmax implements Layer {
  forward(input: Matrix) {
    return softmax(input);
  }
}
