import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class Mlp {

    private double[][] wh, wo;
    private double ni;
    private int quantIn, quantOut, quantH;

    public Mlp(int quantIn, int quantOut, int quantIntermN, double ni){//quantInterN = quantidade interna de neuronios
        this.wh = matrizAleatoria(quantIn, quantIntermN);
        this.wo = matrizAleatoria(quantIntermN, quantOut);
        this.ni = ni;
        this.quantIn = quantIn;
        this.quantOut = quantOut;
        this.quantH = quantIntermN;
    }

    public double[] treinar(double[] xIn, double[] y){
        double[] vetEntradas = concVet(xIn, new double[]{1});

        //calcular o valor de H
        double[] h = new double[quantH + 1];
        for (int j = 0; j < quantH; j++){
            for (int i = 0; i <vetEntradas.length; i++){
                h[j] += vetEntradas[i] * wh[i][j];
            }
            h[j] = 1 / (1 + Math.exp(-h[j]));
        }
        h[quantH] = 1;

        double o[] = new double[quantOut];
        for (int j = 0; j < o.length; j++){
            for (int i = 0; i < h.length; i++){
                o[j] += h[i] * wo[i][j];
            }
            o[j] = 1 / (1 + Math.exp(-o[j]));
        }

        double[] deltaO = new double[quantOut];
        for (int j = 0; j < o.length; j++){
            int sinal = 1;
            if(y[j] - o[j] < 0)
                sinal = -1;

            deltaO[j] = o[j] * (1 - o[j]) * Math.pow((y[j] - o[j]), 2) * sinal;//Elevar ao quadrado
        }

        double[] deltaH = new double[quantH];
        double soma;
        for (int i = 0; i < deltaH.length; i++){
            soma = 0;
            for (int j = 0; j < o.length; j++){
                soma += deltaO[j] * wo[i][j];
            }
            deltaH[i] = h[i] * (1 - h[i]) * soma;
        }

        //ajuste de pesos da matriz wo
        for (int i = 0; i <quantH + 1; i++) {
            for (int j = 0; j < quantOut; j++) {
                wo[i][j] = wo[i][j] + this.ni * deltaO[j] * h[i];
            }
        }

        //ajuste de pesos da matriz wh
        for (int i = 0; i < vetEntradas.length ; i++) {
            for (int j = 0; j < quantH ; j++) {
                wh[i][j] = wh[i][j] + this.ni * deltaH[j] * vetEntradas[i];
            }
        }
        return o;
    };

    public double[] testar(double[] xIn, double[] y){
        double[] vetEntradas = concVet(xIn, new double[]{1});

        //calcular o valor de H
        double[] h = new double[quantH + 1];
        for (int j = 0; j < quantH; j++){
            for (int i = 0; i <vetEntradas.length; i++){
                h[j] += vetEntradas[i] * wh[i][j];
            }
            h[j] = 1 / (1 + Math.exp(-h[j]));
        }
        h[quantH] = 1;

        double o[] = new double[quantOut];
        for (int j = 0; j < o.length; j++){
            for (int i = 0; i < h.length; i++){
                o[j] += h[i] * wo[i][j];
            }
            o[j] = 1 / (1 + Math.exp(-o[j]));
        }

        return o;
    };

    private double[] concVet(double[] vetUm, double[] vetDois){
        int tamanho = vetUm.length + vetDois.length;
        double[] vetAux = new double[tamanho];

        System.arraycopy(vetUm, 0, vetAux, 0, vetUm.length);
        System.arraycopy(vetDois,0, vetAux, vetUm.length, vetDois.length);

        return  vetAux;
    };
    private double[][] matrizAleatoria(int quantIn, int quantOut){
        //Random r = new Random();
        Random r = ThreadLocalRandom.current();
        double min = -0.3;
        double max = 0.3;
        double matriz[][] = new double[quantIn + 1][quantOut];

        for (int i = 0; i < quantIn + 1 ; i++){
            for (int j = 0; j < quantOut; j++){
                //matriz[i][j] = min + (max - min) * r.nextDouble();
                matriz[i][j] = ((Math.random() * 0.3) - 0.3);
            }
        }

        return matriz;
    };
}
