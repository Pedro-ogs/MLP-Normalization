import java.io.*;
import java.util.*;

public class MlpMain {

    public static void main(String[] args) {
        List<Data> baseTreino = new ArrayList<>();
        List<Data> baseTeste = new ArrayList<>();
        splitData(baseTreino, baseTeste);


        Mlp perceptron = new Mlp(7, 1, 5, 0.01);
        for (int e = 0; e < 3000; e++) {
            double erroEpoca = 0;
            int erroClassificacaoEpoca = 0;
            for (Data data : baseTreino) {
                double[] x = data.getInput();
                double[] y = data.getOutput();
                double[] o = perceptron.treinar(x, y);

                double erroAmostra = 0;//valor do somatorio
                double erroClassificacao = 0;
                for (int i = 0; i < y.length; i++) {
                    erroAmostra += Math.abs(y[i] - o[i]);

                    if (o[i] >= 0.5) {
                        erroClassificacao += Math.abs(y[i] - 0.995);
                    } else {
                        erroClassificacao += Math.abs(y[i] - 0.005);
                    }
                }
                if(erroClassificacao > 0)
                    erroClassificacao = 1;

                erroClassificacaoEpoca += erroClassificacao;
                erroEpoca += erroAmostra;
            }

            String treino =  "- erro TREINO: " + erroEpoca + " - erro de classificação TREINO: " + erroClassificacaoEpoca;

            erroEpoca = 0;
            erroClassificacaoEpoca = 0;
            for (Data data : baseTeste) {
                double[] x = data.getInput();
                double[] y = data.getOutput();
                double[] o = perceptron.testar(x, y);

                double erroAmostra = 0;//valor do somatorio
                double erroClassificacao = 0;
                for (int i = 0; i < y.length; i++) {
                    erroAmostra += Math.abs(y[i] - o[i]);

                    if (o[i] >= 0.5) {
                        erroClassificacao += Math.abs(y[i] - 0.995);
                    } else {
                        erroClassificacao += Math.abs(y[i] - 0.005);
                    }
                }
                if(erroClassificacao > 0)
                    erroClassificacao = 1;

                erroClassificacaoEpoca += erroClassificacao;
                erroEpoca += erroAmostra;

            }

            String teste =  " - erro TESTE: " + erroEpoca + " - erro de classificação TESTE: " + erroClassificacaoEpoca;

            System.out.println("A epoca: " + e + treino + teste );

        }
    }

    public static List<Data> readData(String fileName){
        List<Data> ecoliData = new LinkedList<>();

        try {
            FileReader fileReader = new FileReader(fileName);
            BufferedReader br = new BufferedReader(fileReader);

            String line;

            while((line = br.readLine()) != null){
                String[] split = line.split("  ");
                double[] input = new double[split.length - 1];
                double[] output = null;

                for (int i = 0; i < input.length; i++){

                    input[i] = Double.parseDouble(split[i]);
                    if (i == input.length - 1) {
                        output = handleOutput(split[i + 1]);
                    }
                }
                if (!Arrays.stream(input).allMatch(x -> x == 0.0) && output != null) {
                    ecoliData.add(new Data(input, output));
                }
            }

            br.close();

        } catch (IOException e){
            System.err.println("Arquivo " + fileName + " não encontrado!");
        }

        return ecoliData;
    }

    private static double[] handleOutput(String value) {//Alterando de 1 e 0 para 0.995 e 0.005
        if (value.equals(" cp"))
            return new double[]{0.995};
        else if (value.equals(" im"))
            return new double[]{0.005};

        return null;
    }

    private static List<String> readDataBase(String fileName){
        List<String> dataBase = new LinkedList<>();

        try {
            FileReader fileReader = new FileReader(fileName);
            BufferedReader br = new BufferedReader(fileReader);

            String line;
            while((line = br.readLine()) != null){
                dataBase.add(line);
            }
            br.close();

        }catch (IOException e) {
            System.err.println("Arquivo " + fileName + " não encontrado!");
        }

        return dataBase;
    }

    public static List<Data> normalization(String name){
        List<Data> dataBase = readData(name);
        List<Data> normalizationBase = new ArrayList<>();
        int normal = 0;

        double[] minInput = new double[dataBase.get(0).getInput().length];
        double[] maxInput = new double[dataBase.get(0).getInput().length];

        for(int i=0; i<minInput.length; i++){
            minInput[i] = dataBase.get(0).getInput()[i];
            maxInput[i] = dataBase.get(0).getInput()[i];
        }

        for(int i=1; i<dataBase.size(); i++){
            for (int j = 0; j < maxInput.length; j++) {
                if(dataBase.get(i).getInput()[j] > maxInput[j])
                    maxInput[j] = dataBase.get(i).getInput()[j];

                if(dataBase.get(i).getInput()[j] < minInput[j])
                    minInput[j] = dataBase.get(i).getInput()[j];
            }
        }

        for(int i=0; i<dataBase.size(); i++){
            double[] newInput = new double[7];
            for (int j = 0; j < maxInput.length; j++) {
                newInput[j] = (dataBase.get(i).getInput()[j] - minInput[j]) / (maxInput[j] - minInput[j]);
            }
            Data newData = new Data(newInput, dataBase.get(i).getOutput());
            normalizationBase.add(newData);
        }

        if(normal == 0){
            normalizationBase = dataBase;
        }

        return normalizationBase;
    }
    private static void createBasesFiles(){
        List<String> dataBase = readDataBase("ecoli.data");
        createFile("baseCP.txt" , "cp", dataBase);//Criar o txt para o resultado CP
        createFile("baseIM.txt" , "im", dataBase);//Criar o txt para o resultado IM
    }

    private static void splitData(List<Data> baseTreino, List<Data> baseTeste) {
        createBasesFiles();
        List<Data> baseCp = normalization("baseCP.txt");

        List<Data> baseIm = normalization("baseIM.txt");

        int size = (int) (baseCp.size() * 0.7);
        Collections.shuffle(baseCp);

        for (int i = 0; i < baseCp.size(); i++){
            if (i < size) {
                baseTreino.add(baseCp.get(i));
            }else {
                baseTeste.add(baseCp.get(i));
            }
        }

        size = (int) (baseIm.size() * 0.7);
        Collections.shuffle(baseIm);

        for (int i = 0; i < baseIm.size(); i++){
            if (i < size) {
                baseTreino.add(baseIm.get(i));
            }else {
                baseTeste.add(baseIm.get(i));
            }
        }

        Collections.shuffle(baseTreino);
        Collections.shuffle(baseTeste);

    }

    private static void createFile(String fileName, String baseClassification, List<String> dataBase) {
        try {
            FileWriter fw = new FileWriter(fileName);
            BufferedWriter bw = new BufferedWriter(fw);

            for (String value : dataBase) {
                if (isBaseType(value, baseClassification)) {
                    bw.write(value);
                    bw.newLine();
                }
            }

            bw.close();
        } catch (IOException e) {
            System.err.println("Erro ao salvar o arquivo: " + e.getMessage());
        }
    }

    private static boolean isBaseType(String value, String baseClassification) {
        String[] array = value.split(" ");

        return array[array.length-1].trim().equals(baseClassification);
    }

}

