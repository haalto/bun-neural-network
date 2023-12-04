import { matrix } from "mathjs";
import { FCLayer } from "./layers/Layer";

const input = matrix([
  [1, 2, 3, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8],
]);
const layer1 = new FCLayer(4, 5);
const layer2 = new FCLayer(5, 2);

layer1.forward(input);
const layer1Output = layer1.getOutput();

if (layer1Output !== null) {
  layer2.forward(layer1Output);
}

const layer2Output = layer2.getOutput();
console.log(layer2Output?.toArray());
