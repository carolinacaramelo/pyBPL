% Demo of fitting a motor program to an image.
function demo_fit_perturbing(num)
    % Parameters
    s= "./processed/image"+num+".png"; %add processed folder when doing inference 
    img = imread(s);
    K = 1; % number of unique parses we want to collect, we will only collect the best parse 
    verbose = true; % describe progress and visualize parse?
    include_mcmc = true; % run mcmc to estimate local variability?
    fast_mode = true; % skip the slow step of fitting strokes to details of the ink?
    
    %if fast_mode
        %fprintf(1,'Fast mode skips the slow step of fitting strokes to details of the ink.\n');
        %warning_mode('Fast mode is for demo purposes only and was not used in paper results.');
    %end
    
    %load('cross_img','img');
    G = fit_motorprograms(img,K,verbose,include_mcmc,fast_mode);
    FolderName = ("/Users/carolinacaramelo/Desktop/results_inference");   % using my directory
    FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
    for iFig = 1:length(FigList)
      FigHandle = FigList(iFig);
      FigName   = strcat(num2str(get(FigHandle, 'Number')), sprintf("image%d",num));
      set(0, 'CurrentFigure', FigHandle);
    %   saveas(FigHandle, strcat(FigName, '.png'));
      saveas(FigHandle, fullfile(FolderName,strcat(FigName, '.png'))); % specify the full path
    end

    close all;
    
    scores = G.scores;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/scores.mat', 'scores');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/scores.mat', 'scores');
    best_parse = G.models{1,1};
    strokes_best = G.models{1,1}.S;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/strokes_best.mat', 'strokes_best');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/strokes_best.mat', 'strokes_best');
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1,1}.S{i,1}.ids;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/cell_ids.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/cell_ids.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1,1}.S{i,1}.pos_token;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/pos_token.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/pos_token.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1,1}.S{i,1}.nsub;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/nsub.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/nsub.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1, 1}.S{i, 1}.R.type;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_type.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_type.mat', 'C');
    end
    
    for i = 1:G.models{1,1}.ns
        C= cell(G.models{1,1}.ns,1);
        A =cell(G.models{1,1}.ns,1);
        B =cell(G.models{1,1}.ns,1);
        D =cell(G.models{1,1}.ns,1);
        disp(i)
        disp(G.models{1, 1}.S{i, 1}.R.type)
       
        if G.models{1, 1}.S{i, 1}.R.type == "unihist"
            C{i,1} = G.models{1, 1}.S{i, 1}.R.gpos;
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_gpos.mat', 'C');
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_gpos.mat', 'C');
            disp("done")
    
        elseif G.models{1, 1}.S{i, 1}.R.type == "mid"
            A{i,1} = G.models{1, 1}.S{i, 1}.R.subid_spot;
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_subid.mat', 'A');
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_subid.mat', 'A');
            B{i,1} = G.models{1, 1}.S{i, 1}.R.ncpt;
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_ncpt.mat', 'B');
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_ncpt.mat', 'B');
            D{i,1} = G.models{1, 1}.S{i, 1}.R.attach_spot;
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_attach_spot.mat', 'D');
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_attach_spot.mat', 'D');
        else
             B{i,1} = G.models{1, 1}.S{i, 1}.R.ncpt;
             save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_ncpt.mat', 'B');
             save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_ncpt.mat', 'B');
             D{i,1} = G.models{1, 1}.S{i, 1}.R.attach_spot;
             save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_attach_spot.mat', 'D');
             save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_attach_spot.mat', 'D');

        end
    end 
    
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1, 1}.S{i, 1}.R.nprev ;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/r_nprev.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/r_nprev.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1, 1}.S{i, 1}.invscales_type;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/invscale_type.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/invscale_type.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1, 1}.S{i, 1}.invscales_token;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/invscale_token.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/invscale_token.mat', 'C');
    end
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        C{i,1} = G.models{1, 1}.S{i, 1}.shapes_token    ;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/shapes_token.mat', 'C');
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/shapes_token.mat', 'C');
    end
    
    
    C= cell(G.models{1,1}.ns,1);
    for i = 1:G.models{1,1}.ns
        if G.models{1, 1}.S{i, 1}.R.type ~= "unihist"
            C{i,1} = G.models{1, 1}.S{i, 1}.R.eval_spot_token;
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/bottomup/eval_spot_token.mat', 'C');
            save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/eval_spot_token.mat', 'C');
        end
    end
    
    blur_sigma = G.models{1,1}.blur_sigma;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/blur_sigma.mat', 'blur_sigma');
    
    epsilon = G.models{1,1}.epsilon;
    save('/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pyBPL/model/epsilon.mat', 'epsilon');
    

end