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
        "csv/angles_measured.csv",
        "csv_output/measurement_notes.csv"
        
rule measurement_notes:
    """
    Extract the measurement notes from the xlsx file.
    """        
    input:
        xls = "../data/Popis_Babice_VSE_13082024.xlsx",
    output:
        "csv_output/measurement_notes.csv"
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

rule create_regressions_static:
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
    shell:
        """
        mkdir -p csv_output
        python {input.script}
        cp ./csv_output/regressions_static.csv {output.csv}
        """

rule synchronization_check:
    """
    Check synchronization between optics and force/inclino/elasto
    Both variants, with full timeline and with detail around the release.
    """
    input:
        script = "plot_probes_inclino_force.py",
    output:
        "../outputs/synchro_optics_inclino.pdf",
        "../outputs/synchro_optics_inclino_detail.pdf"
    shell:
        """
        rm -rf ../temp/* || true
        mkdir -p ../temp/optics_with_inclino
        python {input.script}
        pdfunite ../temp/optics_with_inclino/*.pdf ../outputs/synchro_optics_inclino.pdf
        rm -rf ../temp/* || true
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
    shell:
        """
        rm -r ../temp/inclino || true
        mkdir -p ../temp/inclino
        echo "Generating PDF files"
        python plot_pull_major_minor.py
        echo "Merge PDF files"
        pdfunite ../temp/inclino/*.pdf {output.pdf}        
        pdfunite ../temp/inclino/*.pdf {output.M01pdf}        
        """