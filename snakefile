rule fft_optics_boxplot:
    """
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

rule fft_elasto_boxplot:
    """
    Draw frequencies grouped by trees with distingushed date and leaf status.
    The probe Elasto is assumed.
    """
    input:
        data = "csv/solara_FFT.csv",
        script = "plot_fft_elasto.py"
    output:
        "../outputs/fft_elasto_boxplot.pdf"
    shell:
        """
        python {input.script}
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