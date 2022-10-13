% Demo of fitting a motor program to an image.
function demo_fit_classification(run,str,num)
    % Parameters
    s= "./image"+num+".png";
    img = imread(s);
    K = 5; % number of unique parses we want to collect, we will only collect the best parse 
    verbose = true; % describe progress and visualize parse?
    include_mcmc = false; % run mcmc to estimate local variability?
    fast_mode = true; % skip the slow step of fitting strokes to details of the ink?
    
    %if fast_mode
        %fprintf(1,'Fast mode skips the slow step of fitting strokes to details of the ink.\n');
        %warning_mode('Fast mode is for demo purposes only and was not used in paper results.');
    %end
    
    %load('cross_img','img');
    G = fit_motorprograms(img,K,verbose,include_mcmc,fast_mode);
    
    struct = G;
    dir = "/Users/carolinacaramelo/Desktop/Thesis_results/PHASE_4_CLASSIFICATION/model_fits/run"+run+str+"_G.mat";
    save(dir, "struct"); %str is _test or _train, run is the number of the run and num is the number of the image
