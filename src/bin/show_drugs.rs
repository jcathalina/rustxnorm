// use rusqlite::{Connection, Result};
// use rusqlite::NO_PARAMS;
// use std::error::Error;
// use std::collections::HashMap;

use ngrammatic::{CorpusBuilder, Pad};
// use csv::Reader;
use std::error::Error;
use serde::Deserialize;
use std::collections::HashMap;

// #[derive(Debug)]
// struct Drug {
//     rxcui: String,
// }

// #[allow(dead_code)]
// fn find_by_rxcui(rxcui: Vec<&str>) -> Result<()> {
//     /// Finds drug related to given RxCUI code from the RxNorm db.
//
//     let conn = Connection::open("databases/rxnorm.db")?;
//
//     let mut stmt = conn.prepare(
//         "SELECT str, tty, rxcui, rxaui FROM rxnconso WHERE rxcui = ? AND lat = \"ENG\"",
//     )?;
//
//     let rxcuis = stmt.query_map(rxcui, |row| {
//         Ok(Drug {
//             rxcui: row.get(0)?,
//         })
//     })?;
//
//     for ui in rxcuis {
//         println!("Found drug {:?}", ui?.rxcui);
//     }
//
//     Ok(())
// }
//
// #[allow(dead_code)]
// fn find_rxcui_by_name(name: Vec<&str>) {
//
//     let conn: Connection = Connection::open("databases/rxnorm.db")?;
//     let rxcuis: HashMap<String, String> = HashMap::new();
//
//
// }

#[derive(Debug, Deserialize)]
struct Drug {
    drug_name: String
}

fn read_csv(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut all_drugnames: Vec<String> = Vec::new();

    for result in reader.deserialize() {
        let record: String = result?;
        all_drugnames.push(record);
    }

    Ok(all_drugnames)
}

fn write_csv(path: &str, data: HashMap<String, String>) -> csv::Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for (drug_name, drug_match) in data {
        writer.write_record(&vec![drug_name, drug_match])?;
    }
    writer.flush()?;
    Ok(())
}


// fn approx_find(name: &str)  {
//
// }

fn main() {

    println!("Building corpus...");
    let mut corpus = CorpusBuilder::new()
        .arity(2)
        .pad_full(Pad::Auto)
        .finish();

    let drug_names: Vec<String> = read_csv("/home/julius/_rp/rustxnorm/data/s1_drug_name_list.csv").unwrap();
    let fda_dict: Vec<String> = read_csv("/home/julius/_rp/rustxnorm/dictionary/fda_dict_refined.csv").unwrap();
    let mut matches: HashMap<String, String> = HashMap::new();

    println!("Populating corpus...");
    for pattern in fda_dict {
        corpus.add_text(&pattern);
    }

    println!("Creating fuzzy match dictionary...");
    for name in drug_names {
        let results = corpus.search(&name, 0.25);
        let top_match = results.first();

        if top_match.is_some() {
            matches.insert(name, top_match.unwrap().clone().text);
        } else {
            matches.insert(name, String::from("NULL"));
        }
    }

    println!("Writing results to csv...");
    write_csv("/home/julius/_rp/rustxnorm/output/s1_fuzzy_matches.csv", matches).unwrap();

    // lookup_rxcui(vec!["3288"]).unwrap()
    // let a: i64 = approx_find("hah");
    // println!("{}", a);
}