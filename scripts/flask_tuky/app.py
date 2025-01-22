from flask import Flask, render_template, request, jsonify, flash
import pandas as pd

app = Flask(__name__)
app.secret_key = 'super secret key'
# Konfigurace
ROWS_PER_PAGE = 100
CSV_FILE = "images.csv"
SERVER = "http://um.mendelu.cz/dynatree//static/public/cache/"

# Načtení dat
def load_images():
    df = pd.read_csv(CSV_FILE)
    df = df[["filename", "valid"]]
    # if "valid" not in df.columns:
    #     df["valid"] = True
    return df

df = load_images()

@app.route("/")
def index():
    page = int(request.args.get("page", 1))
    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    paginated_data = df.iloc[start:end]
    total_pages = (len(df) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE

    return render_template(
        "index.html",
        images=paginated_data.iterrows(),
        page=page,
        total_pages=total_pages, 
        server = SERVER
    )

@app.route("/toggle_like", methods=["POST"])
def toggle_like():
    global df
    image_id = int(request.json.get("image_id"))
    if image_id in df.index:
        df.at[image_id, "valid"] = not bool(df.at[image_id, "valid"])
        df.to_csv(CSV_FILE, index=False)  # Uložení změn
        return jsonify({"status": "success", "valid": bool(df.at[image_id, "valid"])})
    return jsonify({"status": "error", "message": "Invalid image ID"}), 400

@app.route("/set_status", methods=["POST"])
def set_status():
    global df
    image_ids = request.json.get("image_ids", [])
    image_ids = [int(i) for i in image_ids]
    new_status = request.json.get("status", True)  # True pro OK, False pro opačný
    for image_id in image_ids:
        if image_id in df.index:
            df.at[image_id, "valid"] = new_status
    df.to_csv(CSV_FILE, index=False)  # Uložení změn
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
