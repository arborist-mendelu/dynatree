rule all:
    input:
        "../outputs/fft_boxplots_for_probes.pdf",
        "../outputs/fft_spectra.zip",
        "../outputs/synchro_optics_inclino.pdf",
        "../outputs/synchro_optics_inclino_detail.pdf",
        "../outputs/fft_spectra_by_measurements.zip",
        "../outputs/fft_spectra_elasto_acc2.zip",
        "../outputs/regressions_static.csv",
        "../outputs/static_pulling_std_RopeAngle100.pdf",
        "../outputs/static_pulling_error_propagation.xlsx",
        "../outputs/anotated_regressions_static.csv",
        "csv/angles_measured.csv",
        "csv_output/measurement_notes.csv",
        "../outputs/static_pull_removed_experiments.zip",
        "../outputs/FFT_csv_tukey.csv",
        "../outputs/fft_boxplots_for_probes_tukey.pdf",
        "../outputs/static_pull_first_versus_other_pulls.html",
        "../outputs/static_pull_major_versus_total.html",
        "../outputs/welch.pdf"

        
rule measurement_notes:
    """
    Extract the measurement notes from the xlsx file.
    """        
    input:
        xls = "../data/Popis_Babice_VSE_13082024.xlsx",
    output:
        "csv_output/measurement_notes.csv"
    conda:
        "dynatree"        
    shell:
        """
        python read_measurements_notes.py
        """
        
rule fft_boxplots:
    """
    Draw frequencies grouped by trees with distingushed date and leaf status.
    The probes Elasto, Inclino, Pt3 are assumed.
    The data are from csv/solara_FFT.csv, i.e. fft peaks are confirmed by a 
    human and is more peaks are present, all are included. This script deals
    only with the basic frequency, however.
    """
    input:
        data = "csv/solara_FFT.csv",
    output:
        pdf = "../outputs/fft_boxplots_for_probes.pdf"
    conda:
        "dynatree"        
    shell:
        """
        python plot_fft_boxplots.py
        """

rule fft_spectra:
    """
    Draw frequencies spectra.
    The data are from csv/solara_FFT.csv, i.e. fft peaks are confirmed by a 
    human and is more peaks are present, all are included. 
    Comments are also included to the image.
    
    Runs the script lib_plot_spectra_for_probe.py to create pdf files as in solara
    app (signal above and FFT below). Adds remark and peak info to the image. 
    """
    input:
        data = "csv/solara_FFT.csv",
    output:
        zip = "../outputs/fft_spectra.zip"
    conda:
        "dynatree"        
    shell:
        """
        rm -rf ../temp_spectra || true
        mkdir ../temp_spectra
        python lib_plot_spectra_for_probe.py
        cd ../temp_spectra
        zip {output.zip} *.pdf
        """
        
rule fft_spectra_combine:
    """
    Combine fft spectra by measurement of similar probes.
    """
    input:
        zip = "../outputs/fft_spectra.zip"
    output:
        by_measurement = "../outputs/fft_spectra_by_measurements.zip",
        elasto = "../outputs/fft_spectra_elasto_acc2.zip"    
    conda:
        "dynatree"        
    shell:
        """
        rm -rf ../temp/spectra_combine || true
        mkdir -p ../temp/spectra_combine 
        cd ../temp/spectra_combine 
        unzip ../{input.zip}
        for prefix in $(ls *.pdf | cut -d'_' -f1,2,3,4  | sort -u); do pdfunite $prefix*.pdf $prefix.pdf; done
        rm *_*_*_*_*.pdf
        zip ../{output.by_measurement} *.pdf
        rm *pdf
        unzip ../{input.zip} *Elasto* *a02_z*
        pdfunite *.pdf ../{output.elasto}
        trees=$(ls *|cut -d_ -f3 | sort | uniq)
        for tree in $(ls *|cut -d_ -f3 | sort | uniq); do echo $tree; pdfunite *_${{tree}}_*.pdf ${{tree}}.pdf; done        
        rm *_*.pdf
        zip  out.zip *.pdf 
        cp out.zip ../{output.elasto}
        """

rule static_pull_create_regressions:
    """
    Find regression coefficients for static pull and pulling phase of the pull-release
    experiment.
    """
    input:
        script = "static_pull.py",
        xls = "../data/Popis_Babice_VSE_13082024.xlsx",
        csv = "csv/intervals_split_M01.csv",
        csv_angles_measured = "csv/angles_measured.csv"
    output:
        csv = "../outputs/regressions_static.csv"
    conda:
        "dynatree"
    log: stdout="logs/static_pull_create_regressions.stdout", stderr="logs/static_pull_create_regressions.stderr"    
    shell:
        """
        mkdir -p csv_output
        python {input.script} > {log.stdout} 2> {log.stderr}
        cp ./csv_output/regressions_static.csv {output.csv}
        """

rule synchronization_check:
    """
    Check synchronization between optics and force/inclino/elasto
    Both variants, with full timeline and with detail around the release.
    """
    input:
        script = "plot_probes_inclino_force.py"
    output:
        "../outputs/synchro_optics_inclino.pdf",
        "../outputs/synchro_optics_inclino_detail.pdf"
    conda:
        "dynatree"        
    shell:
        """
        rm -rf ../temp/optics_with_inclino || true
        mkdir -p ../temp/optics_with_inclino
        python {input.script}
        pdfunite ../temp/optics_with_inclino/*.pdf ../outputs/synchro_optics_inclino.pdf
        rm -rf ../temp/optics_with_inclino || true
        mkdir -p ../temp/optics_with_inclino
        python {input.script} --release_detail
        pdfunite ../temp/optics_with_inclino/*.pdf ../outputs/synchro_optics_inclino_detail.pdf
        """
        
rule fft_optics_boxplot:
    """
    PROBABLY OBSOLETE, see fft_boxplots
    Draw frequencies grouped by trees with distingushed date and leaf status.
    The probe Pt3 is assumed.
    """
    input:
        data = "results/fft.csv",
        script = "plot_fft.py"
    output:
        "../outputs/fft_optics_boxplot.pdf"
    conda:
        "dynatree"        
    shell:
        """
        python {input.script}
        """

rule RopeAngle_100_std:
    """
    Nakresli smerodatne odchylky veliciny RopeAngle(100) v natahovaci fazi
    tahove staticke zkousky. Ocekavani, ze smerodatne odchylky budou velke se
    nevyplnilo, protoze nektere dny jsou OK. Asi prejit ke koeficientu determinace
    mezi uhlem a silou. Je mizerny i pro mereni, ktere ma malou smerodatnou 
    odchylku v RopeAngle(100).
    """
    input:
        script = "static_pull_analyze_Rope100.py"
    output:
        img = "../outputs/static_pulling_std_RopeAngle100.pdf"
    conda:
        "dynatree"        
    shell:
        """
        python {input.script}
        """
        
rule static_pulling_error_propagation:
    """'
    Get parital derivatives of slopes in momentum-angle regressions graphs with
    respect to angle. Shows that the regression coefficient is not influenced
    by the rope angle. In particular, one degree change in angle gives 0.7% 
    change in the slopes for BK01 (high angle, 20 degrees) and 0.4% change in 
    the slopes for BK04 (small angle, 10 degrees).
    """
    input:
        script = "static_pull_error_propagation.py"
    output:
        table = "../outputs/static_pulling_error_propagation.xlsx"
    conda:
        "dynatree"        
    shell:
        """
        python {input.script}
        """
    
rule angle_from_measurement:
    """
    Extract angle values from "../data/Popis_Babice_VSE_13082024.xlsx"
    Used by static pulling.
    """
    input:
        xls = "../data/Popis_Babice_VSE_13082024.xlsx",
        script = "static_pull_read_tabulka.py"
    output:
        csv = "csv/angles_measured.csv"
    conda:
        "dynatree"        
    shell:
        """
        mkdir -p csv_output
        python {input.script}
        """

rule plot_pull_major_minor:
    """
    Plots pull data labeled as BlueMajor and BlueMinor etc. 
    Useful to check if the major a minor axes are properly
    recognised. 
    """
    output:
        pdf = "../outputs/pull_major_minor_check.pdf",
        M01pdf = "../outputs/pull_major_minor_check_M01.pdf"        
    conda:
        "dynatree"        
    shell:
        """
        rm -r ../temp/inclino || true
        mkdir -p ../temp/inclino
        echo "Generating PDF files"
        python plot_pull_major_minor.py
        echo "Merge PDF files"
        pdfunite ../temp/inclino/*.pdf {output.pdf}        
        pdfunite ../temp/inclino/*M01.pdf {output.M01pdf}        
        """

rule static_pull_regressions_anotate:
    """
    Merge data from pull
    """        
    input:
        "../outputs/regressions_static.csv", 
        "csv/static_fail.csv",
        "static_pull_anotatte_regressions.py"
    conda:
        "dynatree"        
    output:
        "../outputs/anotated_regressions_static.csv"
    shell:
        """
        python static_pull_anotatte_regressions.py
        """

rule static_pull_plot_failed:
    """
    """
    input: 
        "../outputs/anotated_regressions_static.csv"
    output: 
        "../outputs/static_pull_removed_experiments.zip"
    conda:
        "dynatree"        
    shell:
        """
        rm -rf ../temp/static_fail_images || true
        mkdir -p ../temp/static_fail_images
        python static_pull_suspicious.py
        cd ../temp/static_fail_images
        zip -r ../{output} *.* 
        """
        
rule fft_all_probes:
    """
    Find fft data (main peak). Also create images in ../temp/fft_tukey/
    """
    input: 
        "csv/FFT_failed.csv"
    output: 
        csv = "../outputs/FFT_csv_tukey.csv",
        zip = "../outputs/FFT_spectra.zip"
    conda:
        "dynatree"        
    shell:
        """
        rm -r ../temp/fft_tukey || true
        mkdir ../temp/fft_tukey
        python lib_FFT.py
        cd ../temp/fft_tukey/
        echo "fft images" > readme
        rm ../{output.zip} || true
        zip ../{output.zip} *
        """
        
rule fft_all_probes_boxplots:
    """
    Draw boxplots from fft
    """
    input: 
        "../outputs/FFT_csv_tukey.csv",
        "csv/FFT_manual_peaks.csv"
    output: 
        "../outputs/fft_boxplots_for_probes_tukey.pdf"
    conda:
        "dynatree"        
    shell:
        """
        python plot_fft_boxplots_tukey.py
        """
    
rule static_1_versus_2_3:
    """
    Compares the slopes from momentum-force diagram for the first pull 
    and the second or third pull.
    """
    input:
        csv = "../outputs/anotated_regressions_static.csv",
        script = "static_pull_first_versus_other_pulls.py"
    output:
        "../outputs/static_pull_first_versus_other_pulls.html"
    conda:
        "dynatree"        
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter nbconvert --to html --execute static_pull_first_versus_other_pulls.ipynb --no-input
        rm static_pull_first_versus_other_pulls.ipynb
        mv static_pull_first_versus_other_pulls.html {output}
        """

rule static_major_verus_total:
    """
    Compares the slopes from momentum-force diagram for the major and total
    angle.
    """
    input:
        csv = "../outputs/anotated_regressions_static.csv",
        script = "static_pull_Major_versus_total.py"
    output:
        "../outputs/static_pull_major_versus_total.html"
    conda:
        "dynatree"        
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter nbconvert --to html --execute static_pull_Major_versus_total.ipynb --no-input
        rm static_pull_Major_versus_total.ipynb
        mv static_pull_Major_versus_total.html {output}
        """

rule welch:
    output:
        "../outputs/welch.pdf"
    input:
        "welch_for_acc.py"
    conda:
        "dynatree"        
    shell:
        """
        rm -r ../temp/welch || true
        mkdir ../temp/welch
        python {input}
        cd ../temp/welch
	montage BK*.png -tile 2x2 -geometry +0+0 welch.pdf
        mv welch.zip ../../outputs
        """