package macetutorial;
import java.io.IOException;

public class ArgumentsSimple {
    public static void main(String[] args)
    {
    	try{
	        MACE em;
	        String file = "../../data/crowd_matrix_arguments_nonans.csv";
	        em = new MACE(file);
	        
	        // default settings
	        int iterations = 50;
	        int restarts = 10;
	        double smoothing = 0.01 / (double) em.numLabels;
	        double threshold = 1.0;
	        String controls = null;
	        String predictionName = "../../output/arguments_mace_predicted_labels.csv";
	        boolean variational = false;
	        double alpha = 0.5;
	        double beta = 0.5;
	
	        // run with configuration
	        em.run(iterations, smoothing, restarts, alpha, beta, variational, controls);
	
	        // write results to files
	        // generate predictions
	        String[] predictions = em.decode(threshold);
	        em.writeArrayToFile(predictions, predictionName, "\n");
	
	        // generate competence scores
	        Object[] competence = new Object[em.numAnnotators];
	        for (int i = 0; i < em.numAnnotators; i++) {
	            competence[i] = em.thetas[i][1];
	        }
	        String competenceName = "../../output/arguments_mace_competence.csv";
	        em.writeArrayToFile(competence, competenceName, "\n");
	
	        // generate entropies
	        double[][] pArray = em.getLabelProbabilities();
	        String pName = "../../output/arguments_mace_probabilities.csv";
	        String[] probabilities = new String[em.numInstances];
	        for (int i=0; i < em.numInstances; i++) {
	        	probabilities[i] = "";
	        	for (int j=0; j < em.numLabels; j++) {
	        		if (j < em.numLabels-1){
	        			probabilities[i] += String.format("%.2f", pArray[i][j]) + ", ";
	        		} else {
	        			probabilities[i] += String.format("%.2f", pArray[i][j]);
	        		}
	        	}
	        }
	        em.writeArrayToFile(probabilities, pName, "\n");
	        
	    }
	    catch (IOException e) {
	        System.out
	                .println("\n*****************************************************************");
	        System.out.println("\tFILE ERROR:");
	        System.out.println("\t" + e.getMessage());
	        System.out.println("*****************************************************************");
	    }
	    catch (IllegalArgumentException e) {
	        System.out
	                .println("\n*****************************************************************");
	        System.out.println("\tARGUMENT ERROR:");
	        System.out.println("\t" + e.getMessage());
	        System.out.println("*****************************************************************");
	    }
    }
}
