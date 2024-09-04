rule fft_optics_boxplot:
    """
    Draw frequencies grouped by trees with distingushed date and leaf status.
    The probe Pt3 is assumed.
    """
    input:
        "results/fft.csv", "plot_fft.py"
    output:
        "../outputs/fft_optics_boxplot.pdf"
    shell:
        "python plot_fft.py"