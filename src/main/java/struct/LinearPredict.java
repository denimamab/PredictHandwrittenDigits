package struct;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;

public class LinearPredict extends Predict {

    public LinearPredict(String pathToModel) {
        super(pathToModel);
    }

    public Prediction predict(File image) {
        Tensor input = convertToMnistArray(image);
        Session s = model.session();
        Tensor result = s.runner()
                .feed("x", input)
                .fetch("y")
                .run().get(0);
        float[] fetched = fetchResult(result);
        Prediction prediction = new Prediction(fetched, getDigit(fetched));
        return prediction;
    }
}
