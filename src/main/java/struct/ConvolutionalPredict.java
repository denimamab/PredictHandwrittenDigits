package struct;

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;

public class ConvolutionalPredict extends Predict {

    public ConvolutionalPredict(String pathToModel) {
        super(pathToModel);
    }

    public Prediction predict(File image) {
        Tensor input = convertToMnistArray(image);
        Session s = model.session();
        Tensor result = s.runner()
                .feed("x", input)
                .feed("keep_prob_dropout", Tensors.create(0.75f))
                .fetch("y")
                .run().get(0);
        float[] fetched = fetchResult(result);
        Prediction prediction = new Prediction(fetched, getDigit(fetched));
        return prediction;
    }
}
