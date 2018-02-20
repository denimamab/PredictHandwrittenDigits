package gui;

import struct.ConvolutionalPredict;
import struct.LinearPredict;
import struct.Predict;
import struct.Prediction;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Board extends JFrame implements ActionListener {
    static String PATH_TO_LINEAR_MODEL = "C:\\Users\\Dee\\Desktop\\Labs\\AI-Lab\\TensorFlow\\Linear\\saved\\models\\linear";
    static String PATH_TO_CONVOLUTIONAL_MODEL = "C:\\Users\\Dee\\Desktop\\Labs\\AI-Lab\\TensorFlow\\Convolutional\\saved\\models\\convolutional";

    JPanel left, drawarea, configuration, right, resultConv, resultLinear;
    DrawArea paint;
    JButton clear, predict;
    JLabel convLab, linLab;
    LinearPredict linearPredict;
    ConvolutionalPredict convolutionalPredict;

    public Board() {
        convolutionalPredict = new ConvolutionalPredict(PATH_TO_CONVOLUTIONAL_MODEL);
        linearPredict = new LinearPredict(PATH_TO_LINEAR_MODEL);

        setLayout(new GridLayout(1, 2));
        setTitle("Hand Written Digits Prediction - Linear and Convolutional neural network model");


        left = new JPanel();
        left.setPreferredSize(new Dimension(650, 650));
        left.setMinimumSize(new Dimension(650, 650));
        left.setMaximumSize(new Dimension(650, 650));
        left.setBorder(BorderFactory.createTitledBorder("Draw Area"));

        left.setLayout(new BoxLayout(left, BoxLayout.Y_AXIS));

        configuration = new JPanel();
        clear = new JButton("Clear");
        clear.addActionListener(this);
        predict = new JButton("Predict");
        predict.addActionListener(this);
        configuration.add(clear);
        configuration.add(predict);
        left.add(configuration);

        drawarea = new JPanel();
        drawarea.setMinimumSize(new Dimension(560, 560));
        drawarea.setMaximumSize(new Dimension(560, 560));
        drawarea.setPreferredSize(new Dimension(560, 560));
        paint = new DrawArea();
        drawarea.add(paint);
        left.add(drawarea);

        add(left);

        right = new JPanel();
        right.setLayout(new GridLayout(1, 2));
        right.setPreferredSize(new Dimension(200, 650));
        right.setMinimumSize(new Dimension(200, 650));
        right.setMaximumSize(new Dimension(200, 650));
        right.setBorder(BorderFactory.createTitledBorder("Prediction results"));
        resultConv = new JPanel();
        resultLinear = new JPanel();
        convLab = new JLabel();
        linLab = new JLabel();
        resultLinear.add(linLab);
        resultConv.add(convLab);

        right.add(resultLinear);
        right.add(resultConv);

        add(right);

        setMinimumSize(new Dimension(800, 500));
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

    }

    public static void main(String[] args) {
        new Board();
    }

    public void actionPerformed(ActionEvent e) {
        JButton b = (JButton) e.getSource();
        if (b == clear) {
            paint.clear();
        } else if (b == predict) {
            Predict predict = linearPredict;
            Prediction prediction = predict.predict(paint.getImage());
            StringBuffer s = new StringBuffer();
            s.append("<html><body>");
            s.append("<h2>Linear</h2> <hr>");
            s.append("Digit predicted <span style='color: green; font-size:18px;'>" + prediction.getDigit() + "</span>");
            s.append("<br>With a percentage of : <b>" + String.format("%.2f", prediction.getClasses()[prediction.getDigit()] * 100) + "</b>");
            s.append("<br> <table border='1' style='1px solid black'>");
            s.append("<tr>" +
                    "<td>Digit</td>" +
                    "<td>Percentage</td>" +
                    "</tr>");
            for (int i = 0; i < 10; i++) {
                s.append("<tr>");
                s.append("<td>" + i + "</td>");
                s.append("<td>" + String.format("%.2f", prediction.getClasses()[i] * 100) + "%</td>");
                s.append("</tr>");
            }
            s.append("</table>");
            s.append("</body></html>");
            linLab.setText(s.toString());
            predict = convolutionalPredict;
            prediction = predict.predict(paint.getImage());
            s = new StringBuffer();
            s.append("<html><body>");
            s.append("<h2>Convolutional</h2> <hr>");
            s.append("Digit predicted <span style='color: green; font-size:18px;'>" + prediction.getDigit() + "</span>");
            s.append("<br>With a percentage of : <b>" + String.format("%.2f", prediction.getClasses()[prediction.getDigit()] * 100) + "</b>");
            s.append("</body></html>");
            convLab.setText(s.toString());

        }
    }
}
