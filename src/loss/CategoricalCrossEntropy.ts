import { Matrix } from "mathjs";
import { Loss } from "./Loss";
import { categoricalCrossEntropy } from "../math/loss";

export class CategoricalCrossEntropy implements Loss {
  calculate(output: Matrix, labels: Matrix) {
    return Number(categoricalCrossEntropy(output, labels));
  }
}
