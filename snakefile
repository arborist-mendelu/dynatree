rule all:
    input:
        "../outputs/fft_boxplots_for_probes.pdf",
        "../outputs/fft_spectra.zip",
        "../outputs/synchro_optics_inclino.pdf",
        "../outputs/synchro_optics_inclino_detail.pdf",
        "../outputs/fft_optics_boxplot.pdf"
        
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
