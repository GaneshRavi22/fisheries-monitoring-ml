package org.fisheries.ml;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.mnist.MnistManager;

public class FishDataFetcher extends BaseDataFetcher{

	private static final long serialVersionUID = 1L;
	
	public static final int NUM_EXAMPLES = 6802;
    public static final int NUM_EXAMPLES_TEST = 752;
    public static final Path TRAIN_DATA_ROOT = Paths.get("C:", "Users", "gravi", "Desktop", "VMs", "python_img_proc_vm", "data", "train");
    public static final Path TEST_DATA_ROOT = Paths.get("C:", "Users", "gravi", "Desktop", "VMs", "python_img_proc_vm", "data", "test");
    
    private boolean train;
    private boolean shuffle;
    private boolean binarize;
    private int[] order;
    private Random rng;

    public FishDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
        if(!dirExists()) {
            throw new IOException("Training or test data directory does not exist!");
        }
        String images;
        if(train){
            images = TRAIN_DATA_ROOT.toFile().getAbsolutePath();
            totalExamples = NUM_EXAMPLES;
        } else {
            images = TEST_DATA_ROOT.toFile().getAbsolutePath();
            totalExamples = NUM_EXAMPLES_TEST;
        }

        man = new MnistManager(images, train);

        numOutcomes = 8;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        if(train){
            order = new int[NUM_EXAMPLES];
        } else {
            order = new int[NUM_EXAMPLES_TEST];
        }
        for( int i=0; i<order.length; i++ ) order[i] = i;
        rng = new Random(rngSeed);
        reset();    //Shuffle order
    }
    
	@Override
	public void fetch(int arg0) {
		
	}
	
	private boolean dirExists(){
		File trainDir = TRAIN_DATA_ROOT.toFile();
		if(!trainDir.exists()) {
			return false;
		}
		File testDir = TEST_DATA_ROOT.toFile();
		if(!testDir.exists()) {
			return false;
		}
        return true;
    }

}
