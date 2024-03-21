import java.util.HashMap;
import java.util.Map;

public class QTable {
    private Map<Integer, double[]> qValues;
    private int previousState;

    public QTable() {
        qValues = new HashMap<>();
        previousState = 0;
    }

    public double[] getQValues() {
        return qValues.getOrDefault(previousState, new double[2]);
    }

    public int getPreviousState() {
        return previousState;
    }

    public void updateQValue(int state, int action, double newValue) {
        double[] stateQValues = qValues.getOrDefault(state, new double[2]);
        stateQValues[action] = newValue;
        qValues.put(state, stateQValues);
    }
}