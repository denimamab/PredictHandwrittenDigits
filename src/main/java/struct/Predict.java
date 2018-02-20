package struct;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import utils.ImageUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public abstract class Predict {
    protected SavedModelBundle model;

    public Predict(String pathToModel) {
        this.model = SavedModelBundle.load(pathToModel, "serve");
    }

    protected Tensor<Float> convertToMnistArray(File image){
        BufferedImage bufferedImage = null;
        try {
            bufferedImage = ImageIO.read(image);
        } catch (IOException e) {
            e.printStackTrace();
        }
        float[] matrix = ImageUtils.toMnistArray(bufferedImage);
        float[][] inputs = new float[1][784];
        inputs[0] = matrix;
        return Tensors.create(inputs);
    }

    protected float[] fetchResult(Tensor result){
        float[][] res = new float[1][10];
        result.copyTo(res);
        return res[0];
    }

    /**
     * Predict method
     * @param image
     */
    public abstract Prediction predict(File image);

    /**
     * Get the predicted digit
     * @param prediction
     * @return
     */
    public int getDigit(float[] prediction){
        int maxi = 0;
        for ( int i=0; i<10; i++) {
            if(prediction[i]>prediction[maxi])
                maxi = i;
        }
        return maxi;
    }
}
