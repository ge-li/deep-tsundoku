import os
import re
<<<<<<< HEAD
=======
from collections import OrderedDict
>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
from pathlib import Path
from typing import Callable, List

import gradio as gr
import torch
from PIL.Image import Image
from gradio import CSVLogger
<<<<<<< HEAD
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src.models.image_segmentation import crop_book_spines_in_image


def main():
    model_inference = BookSpineReader()
    frontend = make_frontend(model_inference.predict)
    frontend.launch()


def make_frontend(fn: Callable[[Image], str]):
=======
from transformers import DonutProcessor

from src.models.image_segmentation import crop_book_spines_in_image
from src.recsys.inference import BookEmbedding
from src.spinereader.titleasin import TextToAsin

STAGED_MODEL_DIRNAME = (
    Path(__file__).resolve().parent.parent / "spinereader" / "artifacts"
)
MODEL_FILE = "traced_donut_model_title_only.pt"


def main():
    detection_inference = BookSpineReader()
    recommendation_inference = BookRecommender()

    tabbed_pages = make_frontend(
        detection_inference.predict, recommendation_inference.recommend
    )

    tabbed_pages.launch(share=True)


def make_frontend(
    detection_fn: Callable[[Image], str],
    recommendation_fn: Callable[[List[str], List[str]], List[str]],
):
>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
    """Creates a gradio.Interface frontend for an image to text function"""
    # List of example images
    images_dir = os.path.join(get_project_root(), "data/images")
    example_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1] in [".jpg", ".jpeg", ".png"]
    ]

<<<<<<< HEAD
    with gr.Blocks() as frontend:
        # TODO: Add example images
        # TODO: Change layout https://gradio.app/controlling_layout/
        image = gr.Image(type="pil", label="Bookshelf")
        run_button = gr.Button("Find books")
        # gr.Examples(examples=example_images, inputs=[gr.Image()])
        output = gr.Textbox(label="Recognized books")
        wrong_prediction_button = gr.Button("Flag wrong prediction 🐞")
        user_feedback = gr.Textbox(interactive=True, label="User feedback")

        # Log user feedback
        flag_button = gr.Button("Correct predictions")
        flagging_callback = CSVLogger()
        flag_components = [image, output, user_feedback]
        flagging_callback.setup(flag_components, "user_feedback")
        flag_method = FlagMethod(flagging_callback)
        flag_button.click(
            flag_method,
            inputs=flag_components,
            outputs=[],
            preprocess=False,
            queue=False,
        )

        # Functionality of buttons
        run_button.click(fn, inputs=image, outputs=output)
        wrong_prediction_button.click(
            lambda model_output: model_output, inputs=output, outputs=user_feedback
        )
=======
    def augmented_detection_fn(image, save_image, candidate_books):
        detected_books_string = detection_fn(image)
        if save_image:
            candidate_books = candidate_books + list(
                OrderedDict.fromkeys(detected_books_string.split("\n"))
            )
        else:
            candidate_books = list(
                OrderedDict.fromkeys(detected_books_string.split("\n"))
            )
        return {
            candidate_book_titles: candidate_books,
            output_box: detected_books_string,
        }

    def augmented_recommendation_fn(candidate_books, liked_books):
        print(liked_books)  # testing that multiple input works
        scored_asins = recommendation_fn(candidate_books, liked_books)
        scored_asins["title"] = candidate_books
        scored_asins.sort_values(by=["score"], inplace=True, ascending=False)

        ordered_candidates = list(scored_asins.title)
        print("these are ordered candidates")
        print(ordered_candidates)
        return {
            candidate_book_titles: candidate_books,
            rec_output_box: "\n".join(ordered_candidates),
        }

    with gr.Blocks() as frontend:
        # candidate_book_titles = gr.State(["the adventures of huckleberry finn", "the great gatsby"]) # for debugging
        candidate_book_titles = gr.State([])  # for debugging

        with gr.Tab("Book Detection"):
            gr.Markdown("# 📚 Deep Tsundoku: bookshelf app for book lovers")
            gr.Markdown(
                "Upload images of your bookshelf to get the list of books it contains"
            )

            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Bookshelf")
                    save_image = gr.Checkbox(label="Save previous detected books?")
                    gr.Examples(
                        examples=example_images,
                        inputs=image,
                    )
                output_box = gr.Textbox(label="Recognized books")

            find_books_button = gr.Button("Find books")
            find_books_button.click(
                augmented_detection_fn,
                inputs=[image, save_image, candidate_book_titles],
                outputs=[candidate_book_titles, output_box],
            )
            # find_books_button.click(detection_fn, inputs=[image], outputs=[output_box])

            gr.Markdown("### Flag  wrong prediction 🐞")
            gr.Markdown(
                "Are the books incorrect? Help us improve our model by correcting our mistakes!"
            )
            detect_user_feedback = gr.Textbox(interactive=True, label="User feedback")

            # Log user feedback
            detect_flag_button = gr.Button("Correct predictions")
            detect_flagging_callback = CSVLogger()
            detect_flag_components = [image, output_box, detect_user_feedback]
            detect_flagging_callback.setup(detect_flag_components, "user_feedback")
            detect_flag_method = FlagMethod(detect_flagging_callback)
            detect_flag_button.click(
                detect_flag_method,
                inputs=detect_flag_components,
                outputs=[],
                preprocess=False,
                queue=False,
            )

        with gr.Tab("Book Recommendation"):
            gr.Markdown("# 📚 Deep Tsundoku: bookshelf app for book lovers")
            gr.Markdown(
                "Tell us some books you like and we will recommend books from the bookshelf"
            )

            with gr.Row():
                with gr.Column():
                    liked_book_titles = gr.Textbox(
                        label="Input books that you like on separate lines", lines=5
                    )
                    gr.Examples(
                        [
                            "crime and punishment dostoevsky",
                            "harry potter and the prisoner",
                        ],
                        inputs=[liked_book_titles],
                    )
                rec_output_box = gr.Textbox(label="Ranked books")

            rec_books_button = gr.Button("Recommend books")
            # rec_books_button.click(test_augmented_recommendation_fn, inputs=[used_letters_var], outputs=[used_letters_var, rec_output_box])
            rec_books_button.click(
                augmented_recommendation_fn,
                inputs=[candidate_book_titles, liked_book_titles],
                outputs=[candidate_book_titles, rec_output_box],
            )

            gr.Markdown("### Flag poor recommendations 🐞")
            gr.Markdown(
                "Are the recommended books not to your liking? Help us improve our model by correcting our mistakes!"
            )
            rec_user_feedback = gr.Textbox(interactive=True, label="User feedback")

            # Log user feedback
            rec_flag_button = gr.Button("Correct predictions")
            rec_flagging_callback = CSVLogger()
            rec_flag_components = [liked_book_titles, rec_output_box, rec_user_feedback]
            rec_flagging_callback.setup(rec_flag_components, "user_feedback")
            rec_flag_method = FlagMethod(rec_flagging_callback)
            rec_flag_button.click(
                rec_flag_method,
                inputs=rec_flag_components,
                outputs=[],
                preprocess=False,
                queue=False,
            )

    # tabbed_pages = gr.TabbedInterface([detection_frontend, recommendation_frontend], ["What books are on the bookshelf?", "What book should I read?"])

>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
    return frontend


class FlagMethod:
    """Copied from gradio's `interface.py` script that mimics the flagging callback"""

    def __init__(self, flagging_callback, flag_option=None):
        self.flagging_callback = flagging_callback
        self.flag_option = flag_option
        self.__name__ = "Flag"

    def __call__(self, *flag_data):
        self.flagging_callback.flag(flag_data, flag_option=self.flag_option)


<<<<<<< HEAD
=======
class BookRecommender:
    """
    Uses the embeddings, candidate book titles (detected from bookshelves), and liked book
    titles (user input) to make recommendations.
    """

    def __init__(self):
        self.recommender = BookEmbedding()
        self.text_to_asin_converter = TextToAsin()

    def recommend(self, candidate_book_titles: List[str], liked_book_titles: str):
        liked_book_list = liked_book_titles.split("\n")

        # TODO: cache conversion results so we don't need to convert every single time
        # maybe use hash table to quickly look up noisy titles we have converted already
        candidate_book_asins = self.text_to_asin_converter.title_to_asin(
            candidate_book_titles
        )
        liked_book_asins = self.text_to_asin_converter.title_to_asin(liked_book_list)
        scored_asins = self.recommender.recommend(
            candidate_book_asins, liked_book_asins
        )

        return scored_asins


>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
class BookSpineReader:
    """
    Crops the image into the book spines and runs each image through an ML model
    that identifies the text in the image.
    """

    def __init__(self):
        self.image_reader_model = ImageReader()

    def predict(self, image) -> str:
        # Identify book spines in images
        book_spines = crop_book_spines_in_image(image, output_img_type="pil")
<<<<<<< HEAD
=======
        print(f"Found {len(book_spines)} book spines")
>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
        # Read text in each book spine
        output = [self.image_reader_model.predict(img) for img in book_spines]
        return self._post_process_output(output)

    @staticmethod
    def _post_process_output(model_output: List[str]) -> str:
<<<<<<< HEAD
        model_output_clean = [s for s in model_output if len(s) > 0]
=======
        model_output_clean = [
            s
            for s in model_output
            if len(s) > 0 and s != "The Official Game Guide to the World"
        ]
>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
        if len(model_output_clean) == 0:
            return "No book found in the image 😞 Make sure the books are stacked vertically"
        else:
            return "\n".join(model_output_clean)


class ImageReader:
    """
    Runs a Machine Learning model that reads the text in an image
    """
<<<<<<< HEAD
    def __init__(self, author=True):
        """Initializes processing and inference models."""
        self.author = author
        self.hf_modelhub_name = "jay-jojo-cheng/donut-cover-author" if self.author else "jay-jojo-cheng/donut-cover"
        self.processor = DonutProcessor.from_pretrained(self.hf_modelhub_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.hf_modelhub_name)
        self.task_prompt = "<s_cover>" if self.author else "<s_cord-v2>"
        
=======

    def __init__(self, model_path=None, author=True):
        """Initializes processing and inference models."""
        self.author = author
        # self.hf_modelhub_name = "jay-jojo-cheng/donut-cover-author" if self.author else "jay-jojo-cheng/donut-cover"
        self.hf_modelhub_name = "jay-jojo-cheng/donut-cover"  # we are now original title-only model 22.10.12
        self.processor = DonutProcessor.from_pretrained(self.hf_modelhub_name)
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        # self.model = VisionEncoderDecoderModel.from_pretrained(self.hf_modelhub_name)
        self.model = torch.jit.load(model_path)

        # self.task_prompt = "<s_cover>" if self.author else "<s_cord-v2>"
        self.task_prompt = (
            "<s_cover>"  # both title-only and title-author use this prompt
        )

>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
    def predict(self, image) -> str:
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

<<<<<<< HEAD
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        outputs = self.model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )
        return self._post_process_output(outputs)

    def _post_process_output(self, outputs) -> str:
        sequence = self.processor.batch_decode(outputs.sequences)[0]
=======
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"  # note that the torchscript was traced for CPU.
        # If we want to do inference on GPU, we need a GPU traced version
        # self.model.to(device)

        outputs = self.model.generate(pixel_values, decoder_input_ids.to(device))

        return self._post_process_output(outputs)

    def _post_process_output(self, outputs) -> str:
        sequence = self.processor.batch_decode(outputs)[0]
>>>>>>> 921d38fb755d016aaf25db6d05c207e8bb0d8984
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(
            r"<.*?>", "", sequence, count=1
        ).strip()  # remove first task start token
        print(f"Prediction: {sequence}")
        return sequence


def get_project_root():
    """Returns the path to the project's root directory: deep-tsundoku"""
    return Path(__file__).parent.parent.parent


if __name__ == "__main__":
    main()
