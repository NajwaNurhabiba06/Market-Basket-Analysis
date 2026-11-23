import io
import math
from itertools import islice

import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


st.set_page_config(
    page_title="Market Basket Analysis - Book-Crossing",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Market Basket Analysis")
st.subheader("Analisis Kombinasi Buku Menggunakan Algoritma FP-Growth")
st.write(
    "Aplikasi ini mengolah dataset Book-Crossing untuk menemukan kombinasi "
    "dua buku yang sering dibaca atau dinilai oleh pengguna yang sama."
)

st.info(
    "Unggah tiga file dataset: **BX-Book-Ratings.csv**, **BX-Books.csv**, "
    "dan **BX-Users.csv**. File dibaca dengan pemisah titik koma (`;`) "
    "dan encoding Latin-1, mengikuti format dataset pada notebook."
)


@st.cache_data(show_spinner=False)
def read_dataset(file_bytes: bytes, dtype_isbn: bool = False) -> pd.DataFrame:
    """Membaca CSV Book-Crossing."""
    dtype = {"ISBN": "string"} if dtype_isbn else None
    return pd.read_csv(
        io.BytesIO(file_bytes),
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        dtype=dtype
    )


def require_columns(df: pd.DataFrame, required: list, dataset_name: str):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom pada {dataset_name} tidak lengkap. "
            f"Kolom yang tidak ditemukan: {', '.join(missing)}"
        )


def format_support(value: float) -> str:
    return f"{value:.6f} ({value * 100:.4f}%)"


with st.sidebar:
    st.header("Pengaturan Analisis")
    min_user_fraction = st.number_input(
        "Proporsi minimum pengguna per ISBN",
        min_value=0.0001,
        max_value=0.1,
        value=0.001,
        step=0.001,
        format="%.4f",
        help=(
            "Digunakan untuk menyaring ISBN yang muncul pada sejumlah "
            "minimum pengguna sebelum FP-Growth dijalankan."
        )
    )
    min_support = st.number_input(
        "Minimum support FP-Growth",
        min_value=0.0001,
        max_value=1.0,
        value=0.005,
        step=0.001,
        format="%.4f",
        help="Semakin kecil nilainya, semakin banyak kombinasi yang mungkin ditemukan."
    )
    show_itemsets = st.checkbox("Tampilkan seluruh frequent itemset", value=False)


col1, col2, col3 = st.columns(3)
with col1:
    ratings_file = st.file_uploader(
        "1. Unggah Book-Ratings",
        type=["csv"],
        key="ratings"
    )
with col2:
    books_file = st.file_uploader(
        "2. Unggah Books",
        type=["csv"],
        key="books"
    )
with col3:
    users_file = st.file_uploader(
        "3. Unggah Users",
        type=["csv"],
        key="users"
    )

if ratings_file and books_file and users_file:
    if st.button("🚀 Jalankan Market Basket Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Membaca dan memproses dataset..."):
                data_br = read_dataset(ratings_file.getvalue(), dtype_isbn=True)
                data_b = read_dataset(books_file.getvalue(), dtype_isbn=True)
                data_u = read_dataset(users_file.getvalue())

                require_columns(data_br, ["User-ID", "ISBN", "Book-Rating"], "Book-Ratings")
                require_columns(data_b, ["ISBN", "Book-Title"], "Books")
                require_columns(data_u, ["User-ID"], "Users")

                # DATA UNDERSTANDING & DATA CLEANING
                st.session_state["data_br"] = data_br
                st.session_state["data_b"] = data_b
                st.session_state["data_u"] = data_u

                # INTEGRASI DATA
                df = pd.merge(data_br, data_b, on="ISBN", how="left")
                df = pd.merge(df, data_u, on="User-ID", how="left")

                # TRANSACTION FORMATION
                transaction_data = df[["User-ID", "ISBN"]].dropna(subset=["User-ID", "ISBN"])
                basket = (
                    transaction_data.groupby("User-ID")["ISBN"]
                    .apply(lambda values: list(pd.unique(values)))
                )
                basket = basket[basket.apply(len) > 0]

                if basket.empty:
                    raise ValueError("Tidak ditemukan transaksi yang dapat dianalisis.")

                # FILTER ISBN BERDASARKAN JUMLAH PENGGUNA
                min_count = max(1, int(math.ceil(min_user_fraction * len(basket))))
                book_counts = (
                    transaction_data.groupby("ISBN")["User-ID"]
                    .nunique()
                )
                frequent_isbns = set(book_counts[book_counts >= min_count].index)

                basket_filtered = basket.apply(
                    lambda books: [isbn for isbn in books if isbn in frequent_isbns]
                )
                basket_filtered = basket_filtered[basket_filtered.apply(len) > 0]

                if basket_filtered.empty:
                    raise ValueError(
                        "Tidak ada transaksi tersisa setelah penyaringan ISBN. "
                        "Coba turunkan proporsi minimum pengguna."
                    )

                # ONE-HOT ENCODING (SPARSE)
                encoder = TransactionEncoder()
                encoded_array = encoder.fit(basket_filtered).transform(
                    basket_filtered,
                    sparse=True
                )
                data_encoded = pd.DataFrame.sparse.from_spmatrix(
                    encoded_array,
                    columns=encoder.columns_
                )

                if data_encoded.shape[1] == 0:
                    raise ValueError("Tidak ada ISBN yang dapat di-encode.")

                # FREQUENT ITEMSET MINING: FP-GROWTH
                frequent_itemsets = fpgrowth(
                    data_encoded,
                    min_support=float(min_support),
                    use_colnames=True
                )

                if frequent_itemsets.empty:
                    raise ValueError(
                        "Tidak ditemukan frequent itemset. "
                        "Coba turunkan nilai minimum support."
                    )

                frequent_itemsets["Jumlah_Buku"] = frequent_itemsets["itemsets"].apply(len)
                pairs = frequent_itemsets[
                    frequent_itemsets["Jumlah_Buku"] == 2
                ].copy()

                # UBAH ITEMSET MENJADI DUA KOLOM ISBN
                if not pairs.empty:
                    pairs["ISBN_1"] = pairs["itemsets"].apply(lambda x: sorted(list(x))[0])
                    pairs["ISBN_2"] = pairs["itemsets"].apply(lambda x: sorted(list(x))[1])

                    # Tambahkan judul buku
                    book_titles = data_b[["ISBN", "Book-Title"]].drop_duplicates("ISBN")
                    pairs = pairs.merge(
                        book_titles.rename(
                            columns={"ISBN": "ISBN_1", "Book-Title": "Judul_Buku_1"}
                        ),
                        on="ISBN_1",
                        how="left"
                    )
                    pairs = pairs.merge(
                        book_titles.rename(
                            columns={"ISBN": "ISBN_2", "Book-Title": "Judul_Buku_2"}
                        ),
                        on="ISBN_2",
                        how="left"
                    )
                    result_pairs = pairs[
                        ["ISBN_1", "Judul_Buku_1", "ISBN_2", "Judul_Buku_2", "support"]
                    ].sort_values("support", ascending=False).reset_index(drop=True)
                else:
                    result_pairs = pd.DataFrame(
                        columns=["ISBN_1", "Judul_Buku_1", "ISBN_2", "Judul_Buku_2", "support"]
                    )

                # SIMPAN HASIL KE SESSION STATE
                st.session_state["analysis"] = {
                    "data_br": data_br,
                    "data_b": data_b,
                    "data_u": data_u,
                    "df": df,
                    "basket": basket,
                    "basket_filtered": basket_filtered,
                    "frequent_isbns": frequent_isbns,
                    "frequent_itemsets": frequent_itemsets,
                    "result_pairs": result_pairs,
                    "min_count": min_count,
                    "encoded_shape": data_encoded.shape,
                }

        except Exception as error:
            st.error(f"Analisis gagal dijalankan: {error}")

if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    data_br = result["data_br"]
    data_b = result["data_b"]
    data_u = result["data_u"]
    frequent_itemsets = result["frequent_itemsets"]
    result_pairs = result["result_pairs"]

    st.divider()
    st.header("Hasil Analisis")

    tab_overview, tab_cleaning, tab_integration, tab_patterns = st.tabs(
        ["Ringkasan", "Data Cleaning", "Integrasi & Transaksi", "Frequent Itemsets"]
    )

    with tab_overview:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Jumlah transaksi", f"{len(result['basket']):,}")
        metric2.metric("ISBN setelah filter", f"{len(result['frequent_isbns']):,}")
        metric3.metric("Frequent itemset", f"{len(frequent_itemsets):,}")
        metric4.metric("Kombinasi 2 buku", f"{len(result_pairs):,}")

        st.write(
            f"Minimum jumlah pengguna per ISBN: **{result['min_count']:,} pengguna**."
        )
        st.write(
            f"Minimum support FP-Growth: **{format_support(min_support)}**."
        )

        if not result_pairs.empty:
            st.subheader("Kombinasi dua buku dengan support tertinggi")
            st.dataframe(
                result_pairs.head(10).style.format({"support": "{:.6f}"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "Belum ditemukan kombinasi dua buku. Coba turunkan minimum support "
                "pada panel Pengaturan Analisis, lalu jalankan analisis kembali."
            )

    with tab_cleaning:
        st.subheader("Informasi Dataset")
        info_rows = []
        for name, frame in [
            ("Book-Ratings", data_br),
            ("Books", data_b),
            ("Users", data_u)
        ]:
            info_rows.append({
                "Dataset": name,
                "Jumlah baris": len(frame),
                "Jumlah kolom": len(frame.columns),
                "Missing value": int(frame.isna().sum().sum()),
                "Duplikasi baris": int(frame.duplicated().sum())
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

        st.subheader("Missing Value per Kolom")
        selected_dataset = st.selectbox(
            "Pilih dataset",
            ["Book-Ratings", "Books", "Users"],
            key="missing_dataset"
        )
        selected_frame = {
            "Book-Ratings": data_br,
            "Books": data_b,
            "Users": data_u
        }[selected_dataset]
        missing_table = (
            selected_frame.isna().sum()
            .rename("Jumlah Missing Value")
            .reset_index()
            .rename(columns={"index": "Kolom"})
        )
        st.dataframe(
            missing_table[missing_table["Jumlah Missing Value"] > 0],
            use_container_width=True,
            hide_index=True
        )

        if "Book-Rating" in data_br.columns:
            zero_ratings = int((data_br["Book-Rating"] == 0).sum())
            st.metric("Rating bernilai 0", f"{zero_ratings:,}")
            st.caption(
                "Pada dataset Book-Crossing, rating 0 merupakan rating implisit "
                "dan tidak otomatis dianggap sebagai missing value."
            )

    with tab_integration:
        st.subheader("Hasil Integrasi Data")
        st.write(
            "Book-Ratings digabungkan dengan Books berdasarkan ISBN, kemudian "
            "digabungkan dengan Users berdasarkan User-ID."
        )
        st.metric("Jumlah baris setelah integrasi", f"{len(result['df']):,}")
        st.dataframe(result["df"].head(20), use_container_width=True)

        st.subheader("Pembentukan Transaksi")
        st.write(
            "Setiap pengguna diperlakukan sebagai satu transaksi yang berisi "
            "daftar ISBN buku yang pernah diberi rating."
        )
        basket_preview = result["basket"].reset_index()
        basket_preview.columns = ["User-ID", "Books"]
        st.dataframe(basket_preview.head(20), use_container_width=True)

        st.write(
            f"Jumlah transaksi setelah filter ISBN: **{len(result['basket_filtered']):,}**"
        )
        st.write(
            f"Ukuran matriks one-hot encoding: **{result['encoded_shape'][0]:,} × "
            f"{result['encoded_shape'][1]:,}**"
        )

    with tab_patterns:
        st.subheader("Frequent Itemsets")
        if show_itemsets:
            display_itemsets = frequent_itemsets.copy()
            display_itemsets["itemsets"] = display_itemsets["itemsets"].apply(
                lambda values: ", ".join(sorted(map(str, values)))
            )
            st.dataframe(
                display_itemsets.sort_values("support", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption(
                "Tabel berikut menampilkan kombinasi dua buku (2-itemsets) "
                "yang memenuhi minimum support."
            )

        if not result_pairs.empty:
            st.dataframe(
                result_pairs.style.format({"support": "{:.6f}"}),
                use_container_width=True,
                hide_index=True
            )
            csv = result_pairs.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Unduh hasil kombinasi dua buku (CSV)",
                data=csv,
                file_name="hasil_kombinasi_dua_buku.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Tidak ada kombinasi dua buku pada nilai minimum support ini.")

    st.caption(
        "Catatan: nilai support menunjukkan proporsi transaksi pengguna yang "
        "memuat kombinasi buku tersebut."
    )

else:
    st.warning("Unggah ketiga dataset, kemudian klik tombol **Jalankan Market Basket Analysis**.")
    with st.expander("Format file yang dibutuhkan"):
        st.markdown(
            "- `BX-Book-Ratings.csv` — minimal memiliki kolom `User-ID`, `ISBN`, `Book-Rating`.\n"
            "- `BX-Books.csv` — minimal memiliki kolom `ISBN`, `Book-Title`.\n"
            "- `BX-Users.csv` — minimal memiliki kolom `User-ID`."
        )
