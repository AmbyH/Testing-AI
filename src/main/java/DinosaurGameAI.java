import org.apache.logging.log4j.core.config.Configurator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DinosaurGameAI {
    private static final int JUMP_KEY = KeyEvent.VK_SPACE;
    private static final int DUCK_KEY = KeyEvent.VK_DOWN;
    private static final int WAIT_TIME_MS = 100;
    private static final int MAX_EPISODES = 10;
    private static final double LEARNING_RATE = 0.1;
    private static final double DISCOUNT_FACTOR = 0.9;
    private static final double EXPLORATION_RATE = 0.3;

    private Robot robot;
    private MultiLayerNetwork neuralNetwork;
    private Random random;

    public DinosaurGameAI() {
        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }
        neuralNetwork = createNeuralNetwork();
        random = new Random();
    }

    public void playGame() {
        List<DataSet> trainingData = new ArrayList<>();

        for (int episode = 0; episode < MAX_EPISODES; episode++) {
            resetGame();
            while (true) {
                if (isObstacleComing()) {
                    int action = chooseAction();
                    performAction(action);
                    updateTrainingData(trainingData, action);
                    trainNeuralNetwork(trainingData);
                }
                wait(WAIT_TIME_MS);
            }
        }
    }

    private MultiLayerNetwork createNeuralNetwork() {
        int numInputs = 2;
        int numOutputs = 2;
        int hiddenLayerSize = 16;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LEARNING_RATE))
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(hiddenLayerSize)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(hiddenLayerSize)
                        .nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        return new MultiLayerNetwork(conf);
    }

    private boolean isObstacleComing() {
        int x = 1400;
        int yLow = 640;
        int yHigh = 515;

        Color pixelColorLow = robot.getPixelColor(x, yLow);
        Color pixelColorHigh = robot.getPixelColor(x, yHigh);

        boolean isLowObstacle = isColorLowObstacle(pixelColorLow);
        boolean isHighObstacle = isColorHighObstacle(pixelColorHigh);

        return isLowObstacle || isHighObstacle;
    }

    private boolean isColorLowObstacle(Color color) {
        if (color.getRed() < 100 && color.getGreen() < 100 && color.getBlue() < 100) {
            return true;
        }
        return false;
    }

    private boolean isColorHighObstacle(Color color) {
        if (color.getRed() > 200 && color.getGreen() < 100 && color.getBlue() < 100) {
            return true;
        }
        return false;
    }

    private int chooseAction() {
        if (random.nextDouble() < EXPLORATION_RATE) {
            return random.nextInt(2);
        } else {
            double[] qValues = predictQValues();
            return qValues[0] > qValues[1] ? 0 : 1;
        }
    }

    private double[] predictQValues() {
        double obstacleHeight = getObstacleHeight();
        double obstacleDistance = getObstacleDistance();
        double red = getColorRed(robot.getPixelColor(1400,635));
        double green = getColorGreen(robot.getPixelColor(1400,635));
        double blue = getColorBlue(robot.getPixelColor(1400,635));

        double[] input = {obstacleHeight, obstacleDistance, red, green, blue};


        INDArray inputArray = Nd4j.create(input);

        INDArray output = neuralNetwork.output(inputArray);

        double[] qValues = output.toDoubleVector();

        return qValues;
    }

    private double getColorRed(Color color) {
        return color.getRed();
    }
    private double getColorGreen(Color color) {
        return color.getGreen();
    }
    private double getColorBlue(Color color) {
        return color.getBlue();
    }

    private double getObstacleHeight() {
        int x = 1400;
        int yLow = 640;
        int yHigh = 515;

        Color pixelColorLow = robot.getPixelColor(x, yLow);
        Color pixelColorHigh = robot.getPixelColor(x, yHigh);

        int obstacleHeight = yLow - yHigh;

        return obstacleHeight;
    }

    private double getObstacleDistance() {
        int playerX = 85;
        int obstacleX = 1400;

        int obstacleDistance = obstacleX - playerX;

        return obstacleDistance;
    }

    private void performAction(int action) {
        if (action == 0) {
            jump();
        } else {
            duck();
        }
    }

    private void updateTrainingData(List<DataSet> trainingData, int action) {
        double obstacleHeight = getObstacleHeight();
        double obstacleDistance = getObstacleDistance();

        double[] input = {obstacleHeight, obstacleDistance};

        double reward;
        boolean isGameOver = isGameOver();

        if (isGameOver) {
            reward = -100;
        } else {
            reward = 1;
        }

        double[] qValues = predictQValues();

        double[] updatedQValues = qValues.clone();
        updatedQValues[action] = reward + DISCOUNT_FACTOR * Math.max(qValues[0], qValues[1]);

        INDArray inputArray = Nd4j.create(input);
        INDArray qValuesArray = Nd4j.create(updatedQValues);

        trainingData.add(new DataSet(inputArray, qValuesArray));
    }

    private boolean isGameOver() {
        Color one = robot.getPixelColor(610,440);
        Color two = robot.getPixelColor(770,440);
        Color three = robot.getPixelColor(945,440);
        return checkGrey(one) && checkGrey(two) && checkGrey(three);
    }
    private boolean checkGrey(Color color) {
        int red = color.getRed();
        int green = color.getGreen();
        int blue = color.getBlue();
        if (red == green && green == blue && blue == 172) {
            return true;
        }
        return false;
    }

    private void trainNeuralNetwork(List<DataSet> trainingData) {
        DataSetIterator iterator = new ListDataSetIterator<>(trainingData);
        neuralNetwork.fit(iterator);
        trainingData.clear();
    }

    private void resetGame() {

        robot.keyPress(JUMP_KEY);
        robot.keyRelease(JUMP_KEY);


        wait(WAIT_TIME_MS);
    }

    private void jump() {
        robot.keyPress(JUMP_KEY);
        robot.keyRelease(JUMP_KEY);
    }

    private void duck() {
        robot.keyPress(DUCK_KEY);
        robot.keyRelease(DUCK_KEY);
    }

    private void wait(int milliseconds) {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        String log4j2ConfigFile = "path/to/log4j2.xml";
        Configurator.initialize(null, log4j2ConfigFile);

        DinosaurGameAI ai = new DinosaurGameAI();
        ai.playGame();
    }
}