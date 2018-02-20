package struct;

public class Prediction {
    protected float[] classes;
    protected int digit;

    public Prediction(float[] classes, int digit) {
        this.classes = classes;
        this.digit = digit;
    }

    public float[] getClasses() {
        return classes;
    }

    public int getDigit() {
        return digit;
    }
}
