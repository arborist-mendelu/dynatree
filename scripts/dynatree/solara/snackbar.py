import solara


snackbar_text = solara.reactive("")
snackbar_color = solara.reactive("error")
snackbar_timeout = solara.reactive(5000)
@solara.component
def snack():
    with solara.v.Snackbar(
            v_model=snackbar_text.value != "",
            timeout=snackbar_timeout.value,
            on_v_model=lambda *_: snackbar_text.set(""),
            left=False,
            right=True,
            top=True,
            color=snackbar_color.value,
    ):
        solara.Markdown(snackbar_text.value,
                        style={"--dark-color-text": "white", "--color-text": "white"})
        solara.Button(icon=True, icon_name="mdi-close", color="white", on_click=lambda: snackbar_text.set(""))

def show_snack(text=None, color=None, timeout=None):
    if color is not None:
        snackbar_color.value = color
    if timeout is not None:
        snackbar_timeout.value = timeout
    if text is not None:
        snackbar_text.value = text
