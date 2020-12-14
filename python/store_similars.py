from scipy.sparse import csr_matrix, load_npz
import calc_similarity as cs


def store_similars(filename: str):
    matches: csr_matrix = load_npz("../output/s1_U_cossim_matrix.npz")

    with open(file=filename, mode='w') as csvfile:
        csvfile.write("index, match_idx, similarity\n")
        for i, match in enumerate(matches):
            for mi, md in zip(match.indices, match.data):
                if md < 0.999:  # ignore self-matches.
                    csvfile.write(f"{i}, {mi}, {md}\n")


cs.generate_csr_matrix(src_path="../data/s1_drug_name_list_unique.csv",
                       dest_path="../output/s1_U_cossim_matrix.npz")
store_similars("../output/cossim_matches.csv")
