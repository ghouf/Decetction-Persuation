"""
Arabic Text Preprocessing and Data Augmentation Module for Persuasion Detection.
Designed for ArAIEval Shared Task (Task 1A and Task 1B).
"""

from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import List, Optional, Union

from dotenv import find_dotenv, load_dotenv
import emoji
import pandas as pd

# Automatically locate and load .env from the project root
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=True)
if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv(), override=True)

# High-frequency media and political emojis mapped to descriptive Arabic terms
EMOJI_ARABIC_MAPPING = {
    '🔴': ' عاجل ',
    '🚨': ' تنبيه عاجل ',
    '💔': ' حزن ',
    '👍': ' إعجاب ',
    '👎': ' استنكار ',
    '✌️': ' نصر ',
    '🔥': ' اشتعال ',
    '⚠️': ' تحذير ',
    '📢': ' إعلان ',
    '📌': ' ملاحظة ',
    '🎯': ' هدف ',
    '⚡': ' عاجل ',
    '😡': ' غضب ',
    '👏': ' تصفيق ',
    '🌹': ' تحية ',
}

# Rhetorical definitions for targeted LLM data augmentation (Task 1B minority classes)
PERSUASION_DEFINITIONS = {
    'Appeal_to_Popularity': 'استخدام حجة أن الفكرة صحيحة لأن الأغلبية تدعمها (مغالطة عربة الموسيقى / الاحتكام للعامة)',
    'Straw_Man': 'تشويه حجة الخصم أو تضخيمها بشكل مغلوط لتسهيل الهجوم عليها ودحضها (رجل القش)',
    'Red_Herring': 'طرح موضوع جانبي غير ذي صلة لتشتيت الانتباه عن القضية الأصلية (الفسيخ الأحمر)',
    'Whataboutism': 'تحويل مسار النقد بتوجيه اتهام مقابل أو إثارة قضية أخرى لتبرير الخطأ (وماذا عن...؟)',
    'Appeal_to_Time': 'الزعم بأن فكرة ما صالحة أو خاطئة بحكم الوقت والسرعة أو لأنها قديمة/جديدة',
    'Guilt_by_Association': 'تشويه فكرة أو شخص بسبب وجود صلة بمجموعة غير مرغوبة (الذنب بالتبعية)',
    'Repetition': 'التكرار المستمر لنفس الفكرة أو العبارة لترسيخها في أذهان الجمهور',
    'Conversation_Killer': 'عبارات قطعية مبتذلة تهدف إلى إيقاف النقاش والتفكير (مثل: هذا هو الواقع، لا فائدة من الكلام)',
    'False_Dilemma-No_Choice': 'حصر المشكلة في خيارين متطرفين فقط دون النظر لبدائل (إما معنا أو ضدنا)',
    'Consequential_Oversimplification': 'الزعم بأن خطوة واحدة ستؤدي حتماً لكارثة متسلسلة غير مثبتة (المنحدر الزلق)',
}


def clean_arabic_text(text: str) -> str:
    """
    Cleans and standardizes Arabic text (both news paragraphs and tweets).
    - Translates 'LINK' and URLs to Arabic 'رابط'
    - Removes 'RT', '@USER', and handles mentions
    - Converts '#hashtag' to regular spaced words
    - Maps emojis to descriptive Arabic words
    - Removes Latin characters and digits (Western 0-9 and Eastern Arabic ٠-٩)
    - Retains only Arabic letters and rhetorical punctuation marks (? ! ؟)
    - Strips tatweel (kashida) and normalizes whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Translate 'LINK' and web URLs to Arabic 'رابط' before Latin removal
    text = re.sub(r'\bLINK\b', 'رابط', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', 'رابط', text)

    # 2. Clean Retweets (RT) and user mentions
    text = re.sub(r'\bRT\b', ' ', text)
    text = re.sub(r'@USER\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)

    # 3. Clean hashtags: remove '#' and replace underscores with space
    text = re.sub(r'#([^\s#]+)', lambda m: ' ' + m.group(1).replace('_', ' ') + ' ', text)

    # 4. Map emojis to Arabic words
    for emo, ar_repr in EMOJI_ARABIC_MAPPING.items():
        if emo in text:
            text = text.replace(emo, ar_repr)
    text = emoji.demojize(text, language='ar')
    text = re.sub(r':([^\s:]+):', lambda m: ' ' + m.group(1).replace('_', ' ') + ' ', text)

    # 5. Remove Latin characters
    text = re.sub(r'[a-zA-Z]+', ' ', text)

    # 6. Remove numbers (Western 0-9 and Eastern Arabic ٠-٩)
    text = re.sub(r'[0-9\u0660-\u0669]+', ' ', text)

    # 7. Retain only Arabic letters, whitespace, and rhetorical punctuation (? ! ؟)
    text = re.sub(r'[^\u0600-\u06FF\s?!؟]', ' ', text)

    # 8. Remove tatweel / kashida (ـ)
    text = re.sub(r'\u0640+', '', text)

    # 9. Normalize multiple punctuation marks
    text = re.sub(r'[!]+', ' ! ', text)
    text = re.sub(r'[\?؟]+', ' ؟ ', text)

    # 10. Normalize consecutive spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_length_outliers(df: pd.DataFrame, max_words: int = 100, min_words: int = 2) -> pd.DataFrame:
    """
    Removes extreme text length outliers (e.g., massive paragraphs > max_words or empty/near-empty texts).
    """
    word_lengths = df['text'].astype(str).str.split().str.len()
    initial_len = len(df)
    filtered_df = df[(word_lengths >= min_words) & (word_lengths <= max_words)].reset_index(drop=True)
    dropped = initial_len - len(filtered_df)
    if dropped > 0:
        print(f"[Outlier Removal] Dropped {dropped} sample(s) with word length outside [{min_words}, {max_words}].")
    return filtered_df


def augment_minority_classes_gpt4o(
    df: pd.DataFrame,
    target_classes: Optional[List[str]] = None,
    samples_per_class: int = 10,
    api_key: Optional[str] = None
) -> pd.DataFrame:

    try:
        from openai import OpenAI
    except ImportError:
        print("[Augmentation] 'openai' package not installed. Skipping augmentation.")
        return df

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        load_dotenv(ENV_PATH, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("[Augmentation] No OpenAI API Key found in arguments or .env. Skipping augmentation.")
        return df

    client = OpenAI(api_key=api_key)

    if target_classes is None:
        target_classes = [
            'Appeal_to_Popularity',
            'Straw_Man',
            'Red_Herring',
            'Whataboutism',
            'Appeal_to_Time',
            'Guilt_by_Association',
            'Conversation_Killer',
            'False_Dilemma-No_Choice',
            'Consequential_Oversimplification'
        ]

    augmented_records = []
    current_id = 90001

    print(f"[Augmentation] Generating {samples_per_class} samples for {len(target_classes)} minority classes...")

    system_prompt = (
        "أنت خبير في اللغويات الحاسوبية وتحليل الخطاب الإعلامي العربي وأساليب الإقناع والدعاية (ArAIEval / SemEval Taxonomy).\n"
        "مهمتك توليد نصوص باللغة العربية (فقرات إخبارية أو تغريدات) توظف أسلوباً إقناعياً محدداً بدقة وبشكل طبيعي جداً.\n"
        "يجب أن تعيد النتيجة بصيغة JSON حصراً وفق البنية المطلوبة."
    )

    for target_label in target_classes:
        definition = PERSUASION_DEFINITIONS.get(target_label, 'أسلوب إقناعي خطابي')
        user_prompt = (
            f"ولد عدد {samples_per_class} نصوص عربية متنوعة (بعضها أسلوب تغريدة وبعضها أسلوب فقرة صحفية) تستخدم الأسلوب التالي:\n"
            f"الأسلوب المستهدف: {target_label}\n"
            f"تعريف الأسلوب: {definition}\n\n"
            "متطلبات التوليد:\n"
            "1. نصوص واقعية مماثلة للأخبار أو منشورات التواصل الاجتماعي حول مواضيع سياسية، اقتصادية، أو اجتماعية.\n"
            "2. توظيف الأسلوب بوضوح وبلا تكلف.\n"
            "3. أعد قائمة كائنات JSON تحت المفتاح 'samples' بحيث يحتوي كل كائن على: 'text', 'type', 'labels' (تشمل الأسلوب المستهدف)."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            data = json.loads(response.choices[0].message.content)
            samples = data.get("samples", [])
            for s in samples:
                raw_text = s.get("text", "")
                cleaned = clean_arabic_text(raw_text)
                if cleaned:
                    augmented_records.append({
                        "id": f"AUG_{current_id}",
                        "text": cleaned,
                        "labels": s.get("labels", [target_label]),
                        "type": s.get("type", "paragraph")
                    })
                    current_id += 1
        except Exception as e:
            print(f"[Augmentation Warning] Failed generating for {target_label}: {e}")

    if augmented_records:
        df_augmented = pd.DataFrame(augmented_records)
        df_merged = pd.concat([df, df_augmented], ignore_index=True)
        print(f"[Augmentation] Successfully added {len(df_augmented)} synthetic samples. Total: {len(df_merged)}")
        return df_merged

    return df


def preprocess_dataset(
    data: Union[pd.DataFrame, str, Path],
    is_train: bool = True,
    remove_outliers: bool = True,
    max_words: int = 100,
    min_words: int = 2,
    augment: bool = False,
    api_key: Optional[str] = None,
    target_classes: Optional[List[str]] = None,
    samples_per_class: int = 10,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    # 1. Load data
    if isinstance(data, (str, Path)):
        df = pd.read_json(data, lines=True, encoding="utf-8")
    else:
        df = data.copy()

    initial_count = len(df)

    # 2. Apply Arabic text cleaning & format harmonization
    df["text"] = df["text"].astype(str).apply(clean_arabic_text)

    # 3. Task 1A label normalization (if present as string 'true' / 'false')
    if "label" in df.columns:
        if df["label"].dtype == object:
            # Normalize string booleans 'true'/'false' -> 1/0 or boolean
            df["label_binary"] = df["label"].astype(str).str.lower().map({"true": 1, "false": 0})

    # 4. Remove length outliers (only on training data to protect dev/test evaluation integrity)
    if is_train and remove_outliers:
        df = remove_length_outliers(df, max_words=max_words, min_words=min_words)

    # 5. Optional GPT-4o data augmentation (only on training data)
    if is_train and augment:
        df = augment_minority_classes_gpt4o(
            df=df,
            target_classes=target_classes,
            samples_per_class=samples_per_class,
            api_key=api_key
        )

    # 6. Optional save to disk
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out_p, orient="records", lines=True, force_ascii=False)
        print(f"[Pipeline] Exported {len(df):,} records to {out_p.name}")

    print(f"[Pipeline Complete] Initial: {initial_count:,} -> Final: {len(df):,} samples.")
    return df
