from streamlit_drawable_canvas import st_canvas
import streamlit as st
from PIL import Image
import torch
from model import MathCNN
from predict import Predict

class Main:
    def __init__(self):
        self.instance = Predict()
        self.segment_and_predict =self.instance.segment_and_predict

    def main(self):
        st.title("🖋️ Calculating Operations with Handwriting")

        # İşlem butonları için session_state başlat:
        if "selected_operation" not in st.session_state:
            st.session_state["selected_operation"] = None

        # Canvas alanı
        canvas_result = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=10,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=200,
            width=700,
            drawing_mode="freedraw",
            key="canvas"
        )

        # İşlem butonları:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("➕"):
                st.session_state["selected_operation"] = "+"
        with col2:
            if st.button("➖"):
                st.session_state["selected_operation"] = "-"
        with col3:
            if st.button("✖️"):
                st.session_state["selected_operation"] = "*"
        with col4:
            if st.button("➗"):
                st.session_state["selected_operation"] = "/"

        operation = st.session_state["selected_operation"]


        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8")).convert("L")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MathCNN(num_classes=10).to(device)
            model.load_state_dict(torch.load("best_digit_model.pt", map_location=device))
            model.eval()

            predictions = self.segment_and_predict(img, model, device)
            for predicted_element in predictions:
                st.write(" **Predicted Number:** ", predicted_element)

        # 🟡 İşlem seçilmişse hesaplama kısmı:
        if operation and canvas_result.image_data is not None:
            if st.button("🔎 Calculate "):
                try:
                    if len(predictions) >= 2:
                        num1 = int(predictions[0])
                        num2 = int(predictions[1])

                        if operation == "+":
                            result = num1 + num2
                        elif operation == "-":
                            result = num1 - num2
                        elif operation == "*":
                            result = num1 * num2
                        elif operation == "/":
                            result = num1 / num2 if num2 != 0 else "∞ (Bölme sıfıra!)"

                        st.success(f"✅ Result: {num1} {operation} {num2} = {result}")
                    else:
                        st.warning("⚠️ Please select 2 numbers!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    instance = Main()
    instance.main()