require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regressions');
const plot = require('node-remote-plot')

let {
  features,
  labels,
  testFeatures,
  testLabels
} = loadCSV('cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 50
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);


plot({
  x: [1, 2, 3, 4],
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});

console.log('R2 is:', r2);