import pandas as pd
import numpy as np
import psutil
import warnings
warnings.filterwarnings('ignore')

def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe division that handles zeros and invalid values efficiently"""
    result = np.full_like(a, np.nan, dtype=np.float64)
    valid_mask = (b != 0) & ~np.isnan(b) & ~np.isinf(b)
    result[valid_mask] = a[valid_mask] / b[valid_mask]
    return result

def calculate_metabolite_ratios(data):
    """Calculate all metabolite ratios from raw metabolomic data"""
    
    # Replace all negative values with 0 in the entire DataFrame
    data = data.map(lambda x: 0 if isinstance(x, (int, float)) and x < 0 else x)
    
    try:
        # Prepare all new columns in a dictionary first
        new_columns = {}
        
        # Acylcarnitines
        new_columns['(C2+C3)/C0'] = (data['C2'] + data['C3']) / data['C0']
        new_columns['CACT Deficiency (NBS)'] = data['C0'] / (data['C16'] + data['C18'])
        new_columns['CPT-1 Deficiency (NBS)'] = (data['C16'] + data['C18']) / data['C0']
        new_columns['CPT-2 Deficiency (NBS)'] = (data['C16'] + data['C18']) / data['C2']
        new_columns['EMA (NBS)'] = data['C4'] / data['C8']
        new_columns['IBD Deficiency (NBS)'] = data['C4'] / data['C2']
        new_columns['IVA (NBS)'] = data['C5'] / data['C2']
        new_columns['LCHAD Deficiency (NBS)'] = data['C16-OH'] / data['C16']
        new_columns['MA (NBS)'] = data['C3'] / data['C2']
        new_columns['MC Deficiency (NBS)'] = data['C16'] / data['C3']
        new_columns['MCAD Deficiency (NBS)'] = data['C8'] / data['C2']
        new_columns['MCKAT Deficiency (NBS)'] = data['C8'] / data['C10']
        new_columns['MMA (NBS)'] = data['C3'] / data['C0']
        new_columns['PA (NBS)'] = data['C3'] / data['C16']
        new_columns['С2/С0'] = data['C2'] / data['C0']
        new_columns['Ratio of Acetylcarnitine to Carnitine'] = data['C2'] / data['C0']
        
        # Calculate sums once to reuse
        sum_AC_OHs = (data['C5-OH'] + data['C14-OH'] + data['C16-1-OH'] + 
                     data['C16-OH'] + data['C18-1-OH'] + data['C18-OH'])
        sum_ACs = (data['C0'] + data['C10'] + data['C10-1'] + data['C10-2'] + 
                  data['C12'] + data['C12-1'] + data['C14'] + data['C14-1'] + 
                  data['C14-2'] + data['C16'] + data['C16-1'] + data['C18'] + 
                  data['C18-1'] + data['C18-2'] + data['C2'] + data['C3'] + 
                  data['C4'] + data['C5'] + data['C5-1'] + data['C5-DC'] + 
                  data['C6'] + data['C6-DC'] + data['C8'] + data['C8-1'])
        
        new_columns['Ratio of AC-OHs to ACs'] = sum_AC_OHs / sum_ACs
        
        СДК = (data['C14'] + data['C14-1'] + data['C14-2'] + data['C14-OH'] + 
               data['C16'] + data['C16-1'] + data['C16-1-OH'] + data['C16-OH'] + 
               data['C18'] + data['C18-1'] + data['C18-1-OH'] + data['C18-2'] + 
               data['C18-OH'])
        ССК = (data['C6'] + data['C6-DC'] + data['C8'] + data['C8-1'] + 
               data['C10'] + data['C10-1'] + data['C10-2'] + data['C12'] + 
               data['C12-1'])
        СКК = (data['C2'] + data['C3'] + data['C4'] + data['C5'] + data['C5-1'] + 
               data['C5-DC'] + data['C5-OH'])
        
        new_columns['СДК'] = СДК
        new_columns['ССК'] = ССК
        new_columns['СКК'] = СКК
        new_columns['Ratio of Medium-Chain to Long-Chain ACs'] = ССК / СДК
        new_columns['Ratio of Short-Chain to Long-Chain ACs'] = СКК / СДК
        new_columns['Ratio of Short-Chain to Medium-Chain ACs'] = СКК / ССК
        new_columns['SBCAD Deficiency (NBS)'] = data['C5'] / data['C0']
        new_columns['SCAD Deficiency (NBS)'] = data['C4'] / data['C3']
        new_columns['Sum of ACs'] = sum_AC_OHs + sum_ACs - data['C0']  # Subtract C0 since it's included in sum_ACs
        new_columns['Sum of ACs + С0'] = sum_AC_OHs + sum_ACs
        new_columns['Sum of ACs/C0'] = (sum_AC_OHs + sum_ACs - data['C0']) / data['C0']
        
        new_columns['Sum of MUFA-ACs'] = (data['C16-1-OH'] + data['C18-1-OH'] + 
                                        data['C10-1'] + data['C12-1'] + 
                                        data['C14-1'] + data['C16-1'] + 
                                        data['C18-1'] + data['C8-1'] + 
                                        data['C5-1'])
        new_columns['Sum of PUFA-ACs'] = data['C10-2'] + data['C14-2'] + data['C18-2']
        new_columns['TFP Deficiency (NBS)'] = data['C16'] / data['C16-OH']
        new_columns['VLCAD Deficiency (NBS)'] = data['C14-1'] / data['C16']
        new_columns['(C6+C8+C10)/C2'] = (data['C6'] + data['C8'] + data['C10']) / data['C2']
        new_columns['2MBG (NBS)'] = data['C5'] / data['C3']
        new_columns['Carnitine Uptake Defect (NBS)'] = (data['C0'] + data['C2'] + data['C3'] + 
                                                       data['C16'] + data['C18'] + 
                                                       data['C18-1']) / data['Citrulline']

        new_columns['C2 / C3'] = data['C2'] / data['C3']
        # NO- and urea cycle
        new_columns['GABR'] = data['Arginine'] / (data['Ornitine'] + data['Citrulline'])
        new_columns['Orn Synthesis'] = data['Ornitine'] / data['Arginine']
        new_columns['AOR'] = data['Arginine'] / data['Ornitine']
        new_columns['ADMA/(Adenosin+Arginine)'] = data['ADMA'] / (data['Adenosin'] + data['Arginine'])
        new_columns["Asymmetrical Arg Methylation"] = data['ADMA'] / data['Arginine']
        new_columns['Symmetrical Arg Methylation'] = data['TotalDMA (SDMA)'] / data['Arginine']
        new_columns['(Arg+HomoArg)/ADMA'] = (data['Arginine'] + data['Homoarginine']) / data['ADMA']
        new_columns['ADMA / NMMA'] = data['ADMA'] / data['NMMA']
        new_columns['NO-Synthase Activity'] = data['Citrulline'] / data['Arginine']
        new_columns['OTC Deficiency (NBS)'] = data['Ornitine'] / data['Citrulline']
        new_columns['Ratio of HArg to ADMA'] = data['Homoarginine'] / data['ADMA']
        new_columns['Ratio of HArg to SDMA'] = data['Homoarginine'] / data['TotalDMA (SDMA)']
        new_columns['Sum of Asym. and Sym. Arg Methylation'] = (data['TotalDMA (SDMA)'] + data['ADMA']) / data['Arginine']
        new_columns['Sum of Dimethylated Arg'] = data['TotalDMA (SDMA)'] + data['ADMA']
        new_columns['Cit Synthesis'] = data['Citrulline'] / data['Ornitine']
        new_columns['CPS Deficiency (NBS)'] = data['Citrulline'] / data['Phenylalanine']
        new_columns['HomoArg Synthesis'] = data['Homoarginine'] / (data['Arginine'] + data['Lysine'])
        new_columns['Ratio of Pro to Cit'] = data['Proline'] / data['Citrulline']

        # Tryptophan metabolism
        new_columns['Kynurenine / Trp'] = data['Kynurenine'] / data['Tryptophan']
        new_columns['Serotonin / Trp'] = data['Serotonin'] / data['Tryptophan']
        new_columns['Trp/(Kyn+QA)'] = data['Tryptophan'] / (data['Kynurenine'] + data['Quinolinic acid'])
        new_columns['Kyn/Quin'] = data['Kynurenine'] / data['Quinolinic acid']
        new_columns['Quin/HIAA'] = 10 
        new_columns['Tryptamine / IAA'] = data['Tryptamine'] / data['Indole-3-acetic acid']
        new_columns['Kynurenic acid / Kynurenine'] = data['Kynurenic acid'] / data['Kynurenine']

        # Amino acids
        new_columns['Asn Synthesis'] = data['Asparagine'] / data['Aspartic acid']
        new_columns['Glutamine/Glutamate'] = data['Glutamine'] / data['Glutamic acid']
        new_columns['Gly Synthesis'] = data['Glycine'] / data['Serine']
        new_columns['GSG Index'] = data['Glutamic acid'] / (data['Serine'] + data['Glycine'])
        new_columns['GSG_index'] = data['Glutamic acid'] / (data['Serine'] + data['Glycine'])
        new_columns['Sum of Aromatic AAs'] = data['Phenylalanine'] + data['Tyrosin']
        new_columns['BCAA'] = data['Summ Leu-Ile'] + data['Valine']
        new_columns['BCAA/AAA'] = (data['Valine'] + data['Summ Leu-Ile']) / (data['Phenylalanine'] + data['Tyrosin'])
        new_columns['Alanine / Valine'] = data['Alanine'] / data['Valine']
        new_columns['DLD (NBS)'] = data['Proline'] / data['Phenylalanine']
        new_columns['MTHFR Deficiency (NBS)'] = data['Methionine'] / data['Phenylalanine']
        
        # Calculate sums once for AA ratios
        sum_non_essential = (data['Alanine'] + data['Arginine'] + data['Asparagine'] + 
                           data['Aspartic acid'] + data['Glutamine'] + 
                           data['Glutamic acid'] + data['Glycine'] + data['Proline'] + 
                           data['Serine'] + data['Tyrosin'])
        sum_essential = (data['Histidine'] + data['Summ Leu-Ile'] + data['Lysine'] + 
                        data['Methionine'] + data['Phenylalanine'] + 
                        data['Threonine'] + data['Tryptophan'] + data['Valine'])
        
        new_columns['Ratio of Non-Essential to Essential AAs'] = sum_non_essential / sum_essential
        new_columns['Sum of AAs'] = sum_non_essential + sum_essential
        new_columns['Sum of Essential Aas'] = sum_essential
        new_columns['Sum of Non-Essential AAs'] = sum_non_essential
        new_columns['Sum of Solely Glucogenic AAs'] = (data['Alanine'] + data['Arginine'] + 
                                                     data['Asparagine'] + data['Aspartic acid'] + 
                                                     data['Glutamine'] + data['Glutamic acid'] + 
                                                     data['Glycine'] + data['Histidine'] + 
                                                     data['Methionine'] + data['Proline'] + 
                                                     data['Serine'] + data['Threonine'] + 
                                                     data['Valine'])
        new_columns['Sum of Solely Ketogenic AAs'] = data['Summ Leu-Ile'] + data['Lysine']
        new_columns['Valinemia (NBS)'] = data['Valine'] / data['Phenylalanine']
        new_columns['Carnosine Synthesis'] = data['Carnosine'] / data['Histidine']
        new_columns['Histamine Synthesis'] = data['Histamine'] / data['Histidine']

        # Betaine_choline metabolism
        new_columns['Betaine/choline'] = data['Betaine'] / data['Choline']
        new_columns['Methionine + Taurine'] = data['Methionine'] + data['Taurine']
        new_columns['DMG / Choline'] = data['DMG'] / data['Choline']
        new_columns['TMAO Synthesis'] = data['TMAO'] / (data['Betaine'] + data['C0'] + data['Choline'])
        new_columns['TMAO Synthesis (direct)'] = data['TMAO'] / data['Choline']
        new_columns['Met Oxidation'] = data['Methionine-Sulfoxide'] / data['Methionine']

        # Vitamins
        new_columns['Riboflavin / Pantothenic'] = data['Riboflavin'] / data['Pantothenic']

        # ADDED: Oncology-specific ratios that were missing
        new_columns['Arg/ADMA'] = data['Arginine'] / data['ADMA']
        new_columns['Arg/Orn+Cit'] = data['Arginine'] / (data['Ornitine'] + data['Citrulline'])
        new_columns['Glutamine/Glutamate'] = data['Glutamine'] / data['Glutamic acid']
        new_columns['Pro/Cit'] = data['Proline'] / data['Citrulline']
        new_columns['Kyn/Trp'] = data['Kynurenine'] / data['Tryptophan']
        new_columns['Trp/Kyn'] = data['Tryptophan'] / data['Kynurenine']
        
        # Arthritis
        new_columns['Phe/Tyr'] = data['Phenylalanine'] / data['Tyrosin']
        new_columns['Glycine/Serine'] = data['Glycine'] / data['Serine']
        # Lungs
        new_columns['C4 / C2'] = data['C4'] / data['C2']
        new_columns['Valine / Alanine'] = data['Valine'] / data['Alanine']
        # Liver
        new_columns['C0/(C16+C18)'] = data['C0'] / (data['C16'] + data['C18'])
        new_columns['(Leu+IsL)/(C3+С5+С5-1+C5-DC)'] = (data['Summ Leu-Ile']) / (data['C3'] + data['C5'] + data['C5-1'] + data['C5-DC'])
        new_columns['Val/C4'] = data['Valine'] / data['C4']
        new_columns['(C16+C18)/C2'] = (data['C16'] + data['C18']) / data['C2']
        new_columns['C3 / C0'] = data['C3'] / data['C0']

         # Convert the dictionary to a DataFrame
        new_data = pd.DataFrame(new_columns)

        # Get columns that exist in both DataFrames
        common_cols = data.columns.intersection(new_data.columns)

        # For overlapping columns, fill NaN in original data with new_data values
        for col in common_cols:
            data[col] = data[col].fillna(new_data[col])

        # Get columns that only exist in new_data
        new_cols_only = new_data.columns.difference(data.columns)

        # Add the new columns
        data = pd.concat([data, new_data[new_cols_only]], axis=1)

        # Drop Group column if it exists
        if 'Group' in data.columns:
            data = data.drop('Group', axis=1)

        # Print NaN information
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            print("Columns with NaN values:")
            for col in nan_columns:
                nan_count = data[col].isna().sum()
                print(f"  - {col}: {nan_count} NaN values")
        else:
            print("No NaN values found in any columns.")
            
        return data

    except Exception as e:
        print(f"Error calculating metabolite ratios: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

def prepare_final_dataframe(risk_params_df: pd.DataFrame, metabolomic_data_with_ratios: pd.DataFrame, ref_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized preparation of final dataframe with metabolite calculations and risk assessments
    """
    
    # Precompute all metabolite values and z-scores in one pass
    metabolites = risk_params_df['Маркер / Соотношение'].unique()
    metabolite_map = {}
    
    for metabolite in metabolites:
        if metabolite in metabolomic_data_with_ratios.columns:
            value = metabolomic_data_with_ratios[metabolite].iloc[0]
            if pd.isna(value) or np.isinf(value):
                metabolite_map[metabolite] = (np.nan, np.nan)
            else:
                value = max(0, value)
                z_score = calculate_zscore(metabolite, value, ref_stats_df)
                metabolite_map[metabolite] = (value, z_score)
        else:
            metabolite_map[metabolite] = (np.nan, np.nan)
    
    # Vectorized assignment using map
    risk_params_df['Patient'] = risk_params_df['Маркер / Соотношение'].map(lambda x: metabolite_map[x][0])
    risk_params_df['Z_score'] = risk_params_df['Маркер / Соотношение'].map(lambda x: metabolite_map[x][1])
    
    # Optimized subgroup score calculation
    def calculate_subgroup_score_fast(group: pd.DataFrame) -> float:
        """Fast subgroup score calculation using vectorized operations"""
        z_scores = group['Z_score'].values
        
        # Vectorized risk calculation
        abs_zscores = np.abs(z_scores)
        risks = np.zeros_like(abs_zscores)
        
        # Use vectorized conditions
        mask_1 = (abs_zscores >= 1.54) & (abs_zscores <= 1.96)
        mask_2 = abs_zscores > 1.96
        
        risks[mask_1] = 1
        risks[mask_2] = 2
        
        # Remove NaN values
        valid_risks = risks[~np.isnan(risks)]
        max_score = len(valid_risks) * 2
        return np.sum(valid_risks) / max_score * 100 if max_score > 0 else 0
    
    # Calculate subgroup scores
    subgroup_scores = risk_params_df.groupby('Категория').apply(calculate_subgroup_score_fast)
    risk_params_df['Subgroup_score'] = risk_params_df['Категория'].map(subgroup_scores)
    
    return risk_params_df

def load_ref_data(ref_data_path: str) -> pd.DataFrame:
    """Optimized reference data loading"""
    ref_stats = pd.read_excel(ref_data_path, header=None, engine='openpyxl')
    ref_stats.columns = ['stat'] + list(ref_stats.iloc[0, 1:])
    ref_stats = ref_stats.drop(0).set_index('stat')
    
    # Vectorized conversion using apply
    ref_stats = ref_stats.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.'), errors='coerce'))
    
    return ref_stats

def calculate_zscore(metabolite: str, value: float, ref_stats: dict) -> float:
    """Optimized z-score calculation for dictionary format"""
    try:
        if metabolite not in ref_stats:
            return np.nan
            
        stats = ref_stats[metabolite]
        mean = stats['mean']
        sd = stats['sd']
        
        if pd.isna(sd) or sd <= 0 or pd.isna(mean):
            return np.nan
            
        return round((value - mean) / sd, 2)
        
    except (KeyError, TypeError):
        return np.nan

def probability_to_score(prob: float, threshold: float) -> float:
    """Optimized probability to score conversion without numba"""
    prob = min(max(prob, 0), 1)
    if prob < threshold:
        score = 4 * prob / threshold
    else:
        score = 4 + 4 * (prob - threshold) / (1 - threshold)
    return 10 - round(score, 0)

def calculate_risks(risk_params_data: pd.DataFrame, metabolic_data_with_ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized risk calculation with parallel processing
    """
    # Remove duplicates efficiently
    metabolic_data_with_ratios = metabolic_data_with_ratios.loc[~metabolic_data_with_ratios.index.duplicated(keep='first')]
    risk_params_data = risk_params_data.loc[~risk_params_data.index.duplicated(keep='first')]
    
    # Process ML models
    disease_pipelines = {
        "CVD": "CVD.pipeline.CVDPipeline",
        "LIVER": "LIVER.pipeline.LIVERPipeline",
        "PULMO": "PULMO.pipeline.PULMOPipeline",
        "RA": "RA.pipeline.RAPipeline",
        "ONCO": "ONCO.pipeline.ONCOPipeline",
    }
    
    results = []
    
    # Process ML models
    for disease_name, pipeline_path in disease_pipelines.items():
        try:
            module_path, class_name = pipeline_path.rsplit('.', 1)
            module = __import__(f"models.{module_path}", fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            pipeline = pipeline_class()
            
            result = pipeline.calculate_risk(metabolic_data_with_ratios.iloc[0])
            results.append(result)
        except Exception as e:
            results.append({
                "Группа_риска": disease_name,
                "Риск-скор": None,
                "Метод оценки": f"ML модель (ошибка: {str(e)})"
            })
    
    # Process parameter-based groups
    ml_only_groups = {"Состояние сердечно-сосудистой системы", "Состояние функции печени", "Оценка пролиферативных процессов"}
    other_groups = set(risk_params_data['Группа_риска'].unique()) - ml_only_groups
    
    if other_groups and len(metabolic_data_with_ratios) > 0:
        # Vectorized value extraction
        metabolite_values = []
        for metabolite in risk_params_data['Маркер / Соотношение']:
            try:
                value = metabolic_data_with_ratios[metabolite].iloc[0]
                if pd.isna(value) or np.isinf(value):
                    metabolite_values.append(np.nan)
                elif value < 0:
                    metabolite_values.append(0)
                else:
                    metabolite_values.append(value)
            except KeyError:
                metabolite_values.append(np.nan)
        
        risk_params_data = risk_params_data.copy()
        risk_params_data['Patient'] = metabolite_values
        
        # Filter invalid values efficiently using boolean indexing
        valid_mask = ~(risk_params_data['Patient'].isin([np.inf, -np.inf]) | risk_params_data['Patient'].isna())
        risk_params_data_valid = risk_params_data[valid_mask].copy()
        
        # Calculate group scores
        for risk_group in other_groups:
            group_data = risk_params_data_valid[risk_params_data_valid['Группа_риска'] == risk_group]
            if len(group_data) == 0:
                continue
            
            group_score = 10 - (group_data['Subgroup_score'].sum() / (len(group_data) * 100) * 10)
            results.append({
                "Группа_риска": risk_group,
                "Риск-скор": np.round(group_score, 0),
                "Метод оценки": "Параметры"
            })
    
    # Create final DataFrame efficiently
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df.sort_values('Группа_риска', inplace=True)
    
    return result_df[['Группа_риска', 'Риск-скор', 'Метод оценки']].reset_index(drop=True)

def is_dash_process(process: psutil.Process) -> bool:
    """Optimized process checking"""
    try:
        cmdline = ' '.join(process.cmdline()).lower()
        return ('main.py' in cmdline or 'dash' in cmdline) and 'python' in process.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return False
