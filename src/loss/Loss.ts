import { Matrix } from "mathjs";

export abstract class Loss {
  abstract calculate(output: Matrix, labels: Matrix): number;
}
