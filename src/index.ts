import { matrix } from "mathjs";
import { FCLayer } from "./layers/FCLayer";
import { NeuralNetwork } from "./network/NeuralNetwork";
import { ReLU } from "./layers/activation/ReLU";
import { Softmax } from "./layers/activation/Softmax";
import { CategoricalCrossEntropy } from "./loss/CategoricalCrossEntropy";

const input = matrix([
  [1, 2, 3, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8],
]);

const labels = matrix([
  [0, 1],
  [1, 0],
  [1, 0],
]);

const neuralNetwork = new NeuralNetwork(new CategoricalCrossEntropy());

neuralNetwork.addLayer(new FCLayer(4, 5));
neuralNetwork.addLayer(new ReLU());
neuralNetwork.addLayer(new FCLayer(5, 2));
neuralNetwork.addLayer(new Softmax());
const { output, lossValue, accuracy } = neuralNetwork.train(input, labels);

console.log(output.toArray());
console.log(labels.toArray());
console.log(lossValue);
console.log(accuracy);
