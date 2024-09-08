rule all:
    input:
        "../outputs/fft_boxplots_for_probes.pdf",
        "../outputs/fft_spectra.zip",
        "../outputs/synchro_optics_inclino.pdf",
        "../outputs/synchro_optics_inclino_detail.pdf",
        "../outputs/fft_spectra_by_measurements.zip",
        "../outputs/fft_spectra_elasto_acc2.zip"           
        
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
