import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from adjustText import adjust_text

# Custom topic labels — mirrored from cli_visualizations.py.
# Duplicate intentionally for per-corpus flexibility.
# Priority in get_topic_label(): (1) topic_info_file CSV, (2) this dict, (3) "Topic {id}" fallback.
custom_labels = {
        -1: "Outlier",
        0: "Ecosystem Ecology",
        1: "Causality in Physics",
        2: "Agriculture, Food Security & Climate Change",
        3: "Philosophy of Causality",
        4: "Energy Consumption & Environmental Causality",
        5: "Brain Connectivity & Granger Causality",
        6: "MR in Cancer & Genetics",
        7: "Structural Equation Modeling",
        8: "Tourism Satisfaction & Sustainable Tourism",
        9: "Technology Acceptance & Online Learning",
        10: "Entrepreneurial Innovation & Knowledge Management",
        11: "PS Methods & ATE Estimation",
        12: "Trade, Economic Growth & Granger Causality",
        13: "Green Innovation & Green Supply Chain Management",
        14: "Inflation, Exchange Rates & Monetary Policy",
        15: "DAG-Based Task Scheduling & Parallel Computing",
        16: "RDDesign in Education & School Achievement",
        17: "Outlier",
        18: "Medicaid/Medicare Expansion using DiD Methods",
        19: "Sports Participation & Motivational Climate",
        20: "Entrepreneurial Intention & Education",
        21: "Academic Achievement, Self-Efficacy & Motivation",
        22: "Brand Image & Consumer Marketing",
        23: "Travel, Urban Mobility & Autonomous Vehicles",
        24: "Obesity & Eating Disorders",
        25: "Outlier",
        26: "Supply Chain Management",
        27: "Employee Performance, Satisfaction & Organizational Culture",
        28: "Construction Project Management & Project Success",
        29: "Directed Acyclic Graphs",
        30: "Mobile Banking & FinTech Adoption",
        31: "Green Purchase Intention & Green Marketing",
        32: "Air Pollution & Environmental Policy using DiD ",
        33: "Alcohol & Substance Use Disorders in Adolescents",
        34: "Literacy, Fluency & Second Language Acquisition",
        35: "Economics of Fertility and Parental Leave",
        36: "Subspace Identification using IV Techniques",
        37: "Gut Microbiome MR Studies",
        38: "Lung Cancer Surgery & Survival",
        39: "Housing Markets",
        40: "Teacher Performance and Job Satisfaction",
        41: "Occupational Safety",
        42: "Electoral Politics & Voter Turnout using RD",
        43: "Organic Food Purchase Intention & Health Consciousness",
        44: "IV Estimation",
        45: "Bayesian Networks for Fault Diagnosis & Root Cause Analysis",
        46: "Internet & Smartphone Addiction in Adolescents",
        47: "COVID-19 Pharmacoepidemiological PS Studies",
        48: "Nurse Job Satisfaction, Burnout & Work Engagement",
        49: "Road Traffic Safety",
        50: "Natural Disasters",
        51: "Telemedicine & Mobile Health Technology Acceptance",
        52: "Orthopedic Surgery: Knee & Hip Arthroplasty Outcomes",
        53: "HIV: Treatment Adherence & Prevention Behaviors",
        54: "Parenting & Child Emotional Regulation",
        55: "Environmental Regulation & Green Finance",
        56: "Labour Market Policies",
        57: "Policing, Violence & Neighborhood Crime Rates",
        58: "Customer Satisfaction & Service Quality",
        59: "Hematological Malignancies",
        60: "Smoking Cessation & Smoking during Pregnancy",
        61: "Prostate Cancer Surgery",
        62: "Tourism-Led Economic Growth & Granger Causality",
        63: "Brand Community & Loyalty",
        64: "Gastric Cancer Surgery & Survival using PS Methods",
        65: "Blockchain & DAG-Based Distributed Ledger Technologies",
        66: "Adverse Drug Reactions & Pharmacovigilance",
        67: "Intimate Partner Violence & Sexual Violence",
        68: "Cardiac Valve Surgery",
        69: "Breast Cancer Surgery & Survival using PS Methods",
        70: "Internal Audit Quality & Fraud Prevention",
        71: "AI Literacy & Generative AI in Education",
        72: "Corporate Social Responsibility",
        73: "Pancreatic Cancer Surgery & Outcomes using PS Methods",
        74: "Sleep Quality, Insomnia & Mental Health",
        75: "Hepatocellular Carcinoma: Treatment & Survival",
        76: "Diabetes Pharmacotherapy",
        77: "Oil Prices, Stock Markets & Granger Causality",
        78: "Air Pollution & Environmental Health Effects",
        79: "Esophageal Cancer Surgery using PS Methods",
        80: "Waste Management & Recycling ",
        81: "Granger Causality Testing & Time Series Analysis",
        82: "MR in GWAS",
        83: "Gene Regulatory Networks & Single-Cell Expression",
        84: "Coronary Artery Disease",
        85: "Sport Sponsorship & Fan Behavior",
        86: "Impulse Buying & Live-Streaming Commerce",
        87: "Petri Nets & Concurrent Systems",
        88: "Financial Literacy & Behavior",
        89: "Epigenetics & Occupational Exposure to Carcinogens",
        90: "Intensive Care, Sepsis & ICU Mortality",
        91: "Pro-Environmental Behavior & Attitudes",
        92: "Dairy & Livestock Production",
        93: "Suicidal Ideation & Prevention in Adolescents",
        94: "Religiosity, Spirituality & Mental Health",
        95: "Quality of Life & Psychosocial Outcomes in Cancer Patients",
        96: "Liver & Kidney Transplant Outcomes using PS Methods",
        97: "Banking Efficiency, Liquidity & Financial Stability",
        98: "Halal Products & Muslim Consumer Behavior",
        99: "Incarceration & Criminal Justice",
        100: "Insulin Signaling & Insulin Resistance",
        101: "Causal Discovery & Causal Graph Learning",
        102: "Event Causality Extraction & Natural Language Processing",
        103: "Chronic Pain & Fibromyalgia ",
        104: "Ischemic Stroke",
        105: "Dialysis Treatment and Mortality using PS Methods",
        106: "Oral Health & Dental Anxiety",
        107: "Customer Loyalty & Service Quality",
        108: "Digital Transformation & Dynamic Capabilities",
        109: "High-Speed Railway & Regional Development using DiD Methods",
        110: "Epidemiological Causal Inference & Public Health",
        111: "Bayesian Network Learning & Graphical Models",
        112: "MR in Neurological & Psychiatric Disorders",
        113: "Leadership Styles & Employee Creativity",
        114: "PTSD ",
        115: "Islamic Banking & FinTech Adoption in Indonesia",
        116: "Pension Reform & Health Effects using RD Designs",
        117: "Spinal Surgery and Outcomes using PS Methods",
        118: "Renewable Energy Adoption & Energy Saving Behavior",
        119: "Patient Satisfaction & Healthcare Service Quality",
        120: "Renal Cell Carcinoma Studies using PS Methods",
        121: "Electric Vehicle Adoption Intention",
        122: "Corporate Governance, CEO Compensation & Firm Performance",
        123: "Career Adaptability, Employability & Career Development",
        124: "Hepatitis, Antiviral Therapy & Risk",
        125: "Implicit Causality in Linguistics",
        126: "Information and Communication Technology",
        127: "Hepatocellular Carcinoma studies using PS Methods",
        128: "Immigration, Migration & Labour Market Integration",
        129: "Thyroid Cancer Surgery and Outcomes using PS Methods",
        130: "RD Design: Methods & Inference",
        131: "Urban Green Space, Parks & Mental Health",
        132: "Corruption, Anti-Corruption Policies & Economic Growth",
        133: "Blockchain Technology Adoption",
        134: "Vaccine Hesitancy & Vaccination Intention",
        135: "Ophthalmic Surgery",
        136: "Working Memory, Executive Function & Cognitive Ability",
        137: "Corporate Social Responsibility & Customer Loyalty",
        138: "Tax Compliance & Tax Evasion",
        139: "Statin Therapy & Cardiovascular Outcomes",
        140: "IoT, Mobile Commerce & Smart Home Adoption",
        141: "Total Quality Management & Organizational Performance",
        142: "Online Shopping, Trust & Repurchase Intention",
        143: "Mindfulness, Gratitude & Psychological Well-being",
        144: "Coronary Stenting Methods and Outcomes",
        145: "Health Expenditure & Population Health",
        146: "Childbirth Delivery Types & Outcomes using PS Methods",
        147: "Agricultural Commodity Prices & Granger Causality",
        148: "Student Satisfaction & University Service Quality",
        149: "Financial Literacy, Investment Decisions & Behavioral Biases",
        150: "Drug-Induced Liver Injury & Causality Assessment",
        151: "Oral Anticoagulants: Warfarin vs. DOACs in Atrial Fibrillation",
        152: "Causal Mediation Analysis & Moderated Mediation",
        153: "Inflammatory Bowel Disease & Biologic Therapies",
        154: "Work Engagement & Organizational Commitment ",
        155: "Antipsychotic Medications & Schizophrenia Treatment",
        156: "Gallbladder & Biliary Surgery",
        157: "International Remittances, Migration & Poverty Reduction",
        158: "IVF, Embryo Transfer & Reproductive Outcomes",
        159: "Fuzzy Cognitive Maps",
        160: "Board Gender Diversity & Firm Performance",
        161: "Atrial Fibrillation: Catheter Ablation",
        162: "Emotional Intelligence, Leadership & Performance",
        163: "Maternal & Neonatal Health: Antenatal Care",
        164: "Human Resource Management & Performance",
        165: "Online Gaming, Gamification & Purchase Intention",
        166: "Head & Neck Cancer: Radiotherapy & Chemotherapy",
        167: "Causal Reasoning in Child Development",
        168: "Work-Family Conflict & Life Balance",
        169: "Accounting Information Systems & Financial Reporting",
        170: "Cannabis Legalization, Medical Use & Psychosis Risk",
        171: "Augmented/Virtual Reality in Tourism",
        172: "Childhood Asthma, Allergies & Atopic Dermatitis",
        173: "Cybersecurity Behavior & Security Compliance",
        174: "Bladder Cancer",
        175: "Social & Political Trust, Civic Engagement",
        176: "IoT Routing Protocols & Sensor Networks",
        177: "Cryptocurrency Markets & Bitcoin Price Volatility",
        178: "Workplace Spirituality & Spiritual Leadership",
        179: "Organizational Agility & Dynamic Capabilities",
        180: "Personality Traits & Psychometric Assessment",
        181: "Donation Behavior & Charitable Giving",
        182: "Outlier",
        183: "Diabetes Self-Management & Glycemic Control",
        184: "Colorectal Cancer Outcomes using PS Methods",
        185: "Dividend Policy, Capital Structure & Firm Value",
        186: "Colorectal Surgery & Outcomes using PS Methods",
        187: "Multiple Sclerosis: Disease-Modifying Therapies",
        188: "Microfinance & Poverty Reduction",
        189: "Migraine ",
        190: "Opioid Prescribing & Opioid Use Disorder",
        191: "Postpartum Depression & Maternal Mental Health",
        192: "E-Government Adoption & Digital Public Services",
        193: "AI Chatbots & Customer Service",
        194: "Student Creativity & Creative Self-Efficacy",
        195: "Elderly Mental Health, Social Participation & Life Satisfaction",
        196: "Environmental, Social, and Governance Disclosure",
        197: "Loneliness, Social Isolation & Depression in Older Adults",
        198: "AI Adoption in Human Resource Management & Recruitment",
        199: "Peacekeeping Operations, Conflict & Political Science",
        200: "Cloud Computing Adoption & SaaS",
        201: "ADHD: Diagnosis & Interventions in Children & Adults",
        202: "Big Data Analytics & Supply Chain Strategy",
        203: "COPD & Respiratory Outcomes",
        204: "Rheumatoid Arthritis: TNF Inhibitors & Methotrexate",
        205: "Luxury Brand Consumption ",
        206: "Online Food Delivery & Customer Satisfaction",
        207: "Lean Manufacturing & Operational Performance",
        208: "Causal Inference in Recommender Systems",
        209: "Renin-Angiotensin System & Antihypertensive Therapy",
        210: "Military Expenditure, Defense Spending & Economic Growth",
        211: "Cardiac Arrest, CPR & Resuscitation Outcomes",
        212: "DAGs in Clinical & Epidemiological Contexts",
        213: "School Bullying & Victimization",
        214: "Advertising Effectiveness & Ad Avoidance",
        215: "Multisensory Perception & Bayesian Causal Inference",
        216: "Marital Satisfaction & Romantic Relationships",
        217: "Cyber Attacks, Intrusion Detection & Malware",
        218: "Gastric Cancer Risk using PS Methods",
        219: "Cervical & Ovarian Cancer Surgery Survival using PS Methods",
        220: "Brain Tumor Surgery & Survival using PS Methods",
        221: "Organizational Citizenship Behavior & Servant Leadership",
        222: "Enterprise Resource Planning Impacts",
        223: "Rectal Cancer Surgery & Outcomes using PS Methods",
        224: "Terrorism & Fear of Terror",
        225: "Music Education & Self-Efficacy",
        226: "Stigma & Help-Seeking Behavior",
        227: "Vaccine Safety & Adverse Events Following Immunization",
        228: "Outlier",
        229: "Materials Science: Alloys, Coatings & Diffusion",
        230: "Online Privacy & Self-Disclosure",
        231: "Academic Procrastination & Self-Regulation",
        232: "Problem Gambling, Gaming Addiction & Loot Boxes",
        233: "Hernia Repair Surgery",
        234: "Solar-Terrestrial Physics & Geomagnetic Storms",
        235: "Volunteering, Volunteer Motivation & Social Capital",
        236: "Family Business Governance & Succession Planning",
        237: "Intellectual Property & Patent Policy",
        238: "Sleep Disorders & MR",
        239: "Diabetes Disease Management Programs using DiD Methods",
        240: "Helicobacter Pylori & Gastric Ulcer Treatment",
        241: "Quantile Regression & Instrumental Variable Quantile Methods",
        242: "Perfectionism: Adaptive vs. Maladaptive",
        243: "Intrahepatic Cholangiocarcinoma Studies using PS Methods",
        244: "Cyberbullying & Adolescent Cyber Victimization",
        245: "COVID-19 Pandemic Policies & Epidemic Control",
        246: "Juvenile Delinquency & Peer Influence",
        247: "Counterfactual Reasoning & Causal Inference",
        248: "Endoscopic Submucosal Dissection (ESD)",
        249: "AI-Driven Personalization & Digital Marketing",
        250: "Religiosity, Secularization & Sociology of Religion",
        251: "Breastfeeding Practices",
        252: "Child Malnutrition, Stunting & Feeding Practices",
        253: "Employee Turnover Intention & Job Satisfaction",
        254: "Service Robots & Robot Acceptance in Hospitality",
        255: "Causal Inference Methods & Big Data",
        256: "Hearing Loss, Tinnitus & Vestibular Disorders",
        257: "Dynamic Panel Data Models: GMM & Instrumental Variables",
        258: "Malaria Prevention & Chemoprevention",
        259: "Food Tourism & Culinary Tourism",
        260: "Legal Causation, Medical Malpractice & Liability",
        261: "Social Anxiety, Anxiety Disorders & Emotion Regulation",
        262: "Fake News, Journalism & Social Media",
        263: "Support Vector Machines, Multiclass Classification & DAGs",
        264: "Organizational Citizenship Behavior & Organizational Justice",
        265: "Vaccine-Associated Adverse Neurological Events",
        266: "Digital Library Services & User Satisfaction",
        267: "Education Expenditure, Economic Growth & Granger Causality",
        268: "Algorithmic, Counterfactual Fairness & Discrimination",
        269: "Schizophrenia, Neurocognition & Social Functioning",
        270: "Innovative Work Behavior & Transformational Leadership",
        271: "Environmental Kuznets Curve & Ecological Footprint",
        272: "Alzheimer's Disease: Neurodegeneration & Amyloid Pathology",
        273: "Intracranial Aneurysm Treatments & Outcomes using PS Methods",
        274: "Colorectal Liver Metastasis Surgery Outcomes using PS Methods",
        275: "Aortic Aneurysm & Endovascular Repair",
        276: "Hume's Philosophy of Causality",
        277: "Sustainable Fashion & Apparel Consumption",
        278: "Bond Graph Modeling & Causality in Physical Systems",
        279: "Digital Leadership & Employee Digital Competence",
        280: "Phylogenetic Networks & Evolutionary Genomics",
        281: "Causality & Econometrics",
        282: "Outlier",
        283: "Vocational Rehabilitation, Compulsory Schooling & Employment",
        284: "Pulmonary Embolism & Venous Thromboembolism",
        285: "Periodontal Disease, Tooth Loss & Obesity",
        286: "Osteoporosis Treatment: Bisphosphonates & Fracture Prevention",
        287: "Tuberculosis Treatment & MDR-TB",
        288: "Wine Tourism & Wine Consumer Behavior",
        289: "Dementia Risk & Alzheimer's Prevention",
        290: "Bariatric Surgery & Weight Loss",
        291: "Causal Representation Learning & Domain Generalization",
        292: "Neural Architecture Search & Deep Neural Networks",
        293: "Kantian Philosophy of Causality",
        294: "Urological Stone Disease: Nephrolithotomy & Ureteroscopy",
        295: "Internet Use, Social Participation & Depression in Elderly",
        296: "Difference-in-Differences: Parallel Trends & Staggered Adoption",
        297: "Teacher Burnout & Academic Stress",
        298: "Venture Capital Investment & Innovation",
        299: "Mergers & Acquisitions: Market Effects & Regulation",
        300: "Gene Ontology & DAG-Based Ontologies"
}


class BibliographicVisualizer:
    """Visualizer for bibliographic network analyses"""

    def __init__(self, data_dir='.', topic_info_file=None):
        self.data_dir = Path(data_dir)
        self.fig_size = (12, 8)
        self.topic_names = None

        # Load topic name mapping if provided
        if topic_info_file:
            self.load_topic_names(topic_info_file)

    def load_data(self, filename):
        """Load CSV data, filtering outlier topics before returning."""
        df = pd.read_csv(self.data_dir / filename)

        if 'topic' not in df.columns:
            return df

        # Layer 1: BERTopic default outlier (topic == -1)
        before = len(df)
        df = df[df['topic'] != -1]
        dropped_bertopic = before - len(df)
        if dropped_bertopic:
            print(f"  Filtered BERTopic outlier (topic=-1): {dropped_bertopic} row(s) removed")

        # Layer 2: Custom "Outlier" labels — union both CUSTOM_LABELS and topic_info_file CSV
        # Normalise to Python int to avoid numpy.int64 vs int isin() mismatches
        outlier_topic_ids = set()
        for source in [custom_labels, self.topic_names or {}]:
            for tid, name in source.items():
                if str(name).strip().lower() == 'outlier':
                    outlier_topic_ids.add(int(tid))
        if outlier_topic_ids:
            before = len(df)
            df = df[~df['topic'].astype(int).isin(outlier_topic_ids)]
            dropped_custom = before - len(df)
            if dropped_custom:
                print(f"  Filtered custom Outlier topics {sorted(outlier_topic_ids)}: {dropped_custom} row(s) removed")

        return df

    def load_topic_names(self, topic_info_file):
        """
        Load topic names from a topic info CSV file.

        Parameters:
        -----------
        topic_info_file : str
            Path to CSV file with 'Topic' and 'Name' columns
        """
        try:
            # Try multiple path resolution strategies
            topic_info_path = None

            # Strategy 1: Check if it's an absolute path
            if Path(topic_info_file).is_absolute():
                topic_info_path = Path(topic_info_file)
            # Strategy 2: Try relative to data_dir
            elif (self.data_dir / topic_info_file).exists():
                topic_info_path = self.data_dir / topic_info_file
            # Strategy 3: Try relative to current working directory
            elif Path(topic_info_file).exists():
                topic_info_path = Path(topic_info_file)
            else:
                # Try one more time - maybe it's relative to cwd
                topic_info_path = Path.cwd() / topic_info_file

            print(f"Attempting to load topic info from: {topic_info_path}")
            topic_info = pd.read_csv(topic_info_path)

            # Handle different possible column names (case-insensitive)
            topic_col = None
            name_col = None

            # Create a mapping of lowercase column names to actual column names
            col_map = {col.lower(): col for col in topic_info.columns}

            # Look for topic column
            for possible_name in ['topic', 'topic_id', 'topicid']:
                if possible_name in col_map:
                    topic_col = col_map[possible_name]
                    break

            # Look for name column
            for possible_name in ['name', 'topic_name', 'topicname', 'representation']:
                if possible_name in col_map:
                    name_col = col_map[possible_name]
                    break

            if topic_col and name_col:
                self.topic_names = dict(zip(topic_info[topic_col], topic_info[name_col]))
                print(f"✓ Loaded {len(self.topic_names)} topic names from {topic_info_file}")
                print(f"  Using columns: '{topic_col}' (ID) and '{name_col}' (Name)")
            else:
                print(f"⚠ Warning: Could not find 'topic' and 'name' columns in {topic_info_file}")
                print(f"   Available columns: {list(topic_info.columns)}")
                print(f"   Found topic_col: {topic_col}, name_col: {name_col}")
        except FileNotFoundError as e:
            print(f"⚠ Warning: Topic info file not found: {topic_info_file}")
            print(f"   Tried path: {topic_info_path}")
        except Exception as e:
            print(f"⚠ Warning: Error loading topic names: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_topic_label(self, topic_id, short=True):
        """
        Get display label for a topic.

        Priority: (1) topic_info_file CSV, (2) built-in CUSTOM_LABELS, (3) "Topic {id}" fallback.

        Parameters:
        -----------
        topic_id : int or str
            Topic ID
        short : bool
            If True, truncate names longer than 40 chars (for axis labels)

        Returns:
        --------
        str : Topic label for display
        """
        name = None
        if self.topic_names and topic_id in self.topic_names:
            name = self.topic_names[topic_id]
        elif topic_id in custom_labels:
            name = custom_labels[topic_id]

        if name is None:
            return f"Topic {topic_id}"

        if short and len(str(name)) > 40:
            return f"{str(name)[:37]}..."
        return str(name)

    def filter_by_prevalence(self, df, top_n=None, min_works=None):
        """
        Filter topics by prevalence (number of works).

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        top_n : int, optional
            Keep only top N most prevalent topics
        min_works : int, optional
            Keep only topics with at least this many works

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        # Find the works column
        work_col = None
        for possible_col in ['n_works', 'total_works', 'Count', 'count']:
            if possible_col in df.columns:
                work_col = possible_col
                break

        if work_col is None:
            print(f"  ⚠ Warning: No works/count column found for prevalence filtering. Available columns: {list(df.columns)}")
            print(f"  Skipping prevalence filter (cannot determine prevalence without works column)")
            return df

        if top_n is not None:
            df = df.nlargest(top_n, work_col)
            print(f"  Filtered to top {top_n} most prevalent topics (by {work_col})")

        if min_works is not None:
            df = df[df[work_col] >= min_works]
            print(f"  Filtered to topics with {work_col} >= {min_works} ({len(df)} topics)")

        return df

    def filter_by_centrality(self, df, metric='mean_pagerank', top_n=None, bottom_n=None, threshold=None):
        """
        Filter topics by centrality measures.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with centrality metrics
        metric : str
            Centrality metric to filter on (default: 'mean_pagerank')
        top_n : int, optional
            Keep only top N topics by centrality
        bottom_n : int, optional
            Keep only bottom N topics by centrality
        threshold : float, optional
            Keep only topics with centrality above this threshold

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, metric)
            print(f"  Filtered to top {top_n} topics by {metric}")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, metric)
            print(f"  Filtered to bottom {bottom_n} topics by {metric}")

        if threshold is not None:
            df = df[df[metric] >= threshold]
            print(f"  Filtered to topics with {metric} >= {threshold} ({len(df)} topics)")

        return df

    def filter_by_influence(self, df, metric='total_external_citations', top_n=None, bottom_n=None, threshold=None):
        """
        Filter topics by influence metrics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with influence metrics
        metric : str
            Influence metric to filter on (default: 'total_external_citations')
            Options: 'total_external_citations', 'citation_balance', 'h_index_proxy'
        top_n : int, optional
            Keep only top N topics by influence
        bottom_n : int, optional
            Keep only bottom N topics by influence
        threshold : float, optional
            Keep only topics with influence above this threshold

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n is not None:
            df = df.nlargest(top_n, metric)
            print(f"  Filtered to top {top_n} topics by {metric}")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, metric)
            print(f"  Filtered to bottom {bottom_n} topics by {metric}")

        if threshold is not None:
            df = df[df[metric] >= threshold]
            print(f"  Filtered to topics with {metric} >= {threshold} ({len(df)} topics)")

        return df

    def filter_by_trends(self, df, top_n=None, bottom_n=None, top_n_changing=None,
                        top_n_each=None, min_slope=None, max_slope=None,
                        growing_only=False, declining_only=False):
        """
        Filter topics by temporal trends.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with trend metrics
        top_n : int, optional
            Keep only top N topics by trend slope (fastest growing)
        bottom_n : int, optional
            Keep only bottom N topics by trend slope (fastest declining)
        top_n_changing : int, optional
            Keep only top N topics by absolute trend slope (most changing in either direction)
        top_n_each : int, optional
            Keep top N fastest growing AND top N fastest declining topics (balanced selection)
        min_slope : float, optional
            Keep only topics with trend_slope >= this value
        max_slope : float, optional
            Keep only topics with trend_slope <= this value
        growing_only : bool
            Keep only topics with positive trend slope
        declining_only : bool
            Keep only topics with negative trend slope

        Returns:
        --------
        pd.DataFrame : Filtered dataframe
        """
        if top_n_changing is not None:
            df = df.reindex(df['trend_slope'].abs().nlargest(top_n_changing).index)
            print(f"  Filtered to top {top_n_changing} most changing topics (by absolute slope)")

        if top_n_each is not None:
            growing = df.nlargest(top_n_each, 'trend_slope')
            declining = df.nsmallest(top_n_each, 'trend_slope')
            df = pd.concat([growing, declining]).drop_duplicates()
            print(f"  Filtered to top {top_n_each} growing + top {top_n_each} declining topics ({len(df)} total)")

        if top_n is not None:
            df = df.nlargest(top_n, 'trend_slope')
            print(f"  Filtered to top {top_n} fastest growing topics")

        if bottom_n is not None:
            df = df.nsmallest(bottom_n, 'trend_slope')
            print(f"  Filtered to bottom {bottom_n} fastest declining topics")

        if min_slope is not None:
            df = df[df['trend_slope'] >= min_slope]
            print(f"  Filtered to topics with trend_slope >= {min_slope} ({len(df)} topics)")

        if max_slope is not None:
            df = df[df['trend_slope'] <= max_slope]
            print(f"  Filtered to topics with trend_slope <= {max_slope} ({len(df)} topics)")

        if growing_only:
            df = df[df['trend_slope'] > 0]
            print(f"  Filtered to growing topics only ({len(df)} topics)")

        if declining_only:
            df = df[df['trend_slope'] < 0]
            print(f"  Filtered to declining topics only ({len(df)} topics)")

        return df
    
    def visualize_centrality_pagerank_network(self, df, output_file='centrality_pagerank_network.png',
                                              works_file=None):
        """
        Visualize PageRank centrality as a hub-and-spoke network.

        Topic nodes are sized by mean_pagerank and positioned centrally.
        If works_file is provided, individual paper nodes fan out radially
        from their topic hub, mirroring the Cambridge Intelligence PageRank layout.

        Parameters
        ----------
        df : pd.DataFrame
            Centrality data with 'topic', 'mean_pagerank', 'n_works' columns.
        output_file : str
            Path to save the PNG.
        works_file : str or None
            Path to a CSV with a 'topic' column (paper-level assignments).
            When provided, paper satellite nodes are drawn around each hub.
        """
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D

        G = nx.Graph()

        # --- Build topic hub nodes ---
        df_sorted = df.sort_values('mean_pagerank', ascending=False).reset_index(drop=True)
        pagerank_vals = df_sorted['mean_pagerank'].values
        pr_min, pr_max = pagerank_vals.min(), pagerank_vals.max()
        pr_range = pr_max - pr_min if pr_max > pr_min else 1.0

        topic_ids = df_sorted['topic'].tolist()
        for _, row in df_sorted.iterrows():
            G.add_node(f"topic_{int(row['topic'])}", node_type='topic',
                       pagerank=row['mean_pagerank'], n_works=row['n_works'])

        # --- Load paper satellite nodes if works_file provided ---
        topic_to_papers = {}
        if works_file is not None:
            works_df = pd.read_csv(works_file)
            if 'topic' in works_df.columns:
                valid_topics = set(topic_ids)
                for tid in topic_ids:
                    papers = works_df[works_df['topic'] == tid].index.tolist()
                    # Cap satellites at 40 per topic for visual clarity
                    if len(papers) > 40:
                        rng = np.random.default_rng(seed=42)
                        papers = rng.choice(papers, size=40, replace=False).tolist()
                    topic_to_papers[tid] = papers
                    for pid in papers:
                        G.add_node(f"paper_{pid}", node_type='paper', topic=tid)
                        G.add_edge(f"topic_{tid}", f"paper_{pid}")

        # --- Two-pass layout ---
        # Pass 1: spring layout on topic nodes only
        topic_node_ids = [f"topic_{t}" for t in topic_ids]
        topic_subgraph = G.subgraph(topic_node_ids)
        hub_pos = nx.spring_layout(topic_subgraph, k=2.5, seed=42)

        pos = dict(hub_pos)

        # Pass 2: radial arc for each topic's paper satellites
        for tid, papers in topic_to_papers.items():
            hub_xy = np.array(hub_pos[f"topic_{tid}"])
            n = len(papers)
            if n == 0:
                continue
            n_works_val = df_sorted.loc[df_sorted['topic'] == tid, 'n_works'].values[0]
            radius = 0.18 + 0.04 * np.sqrt(n_works_val)
            for j, pid in enumerate(papers):
                angle = 2 * np.pi * j / n
                pos[f"paper_{pid}"] = hub_xy + radius * np.array([np.cos(angle), np.sin(angle)])

        # --- Colours: topics ranked by PageRank using viridis ---
        cmap = plt.cm.PuBuGn
        # iterrows() returns float64 for int columns in mixed-dtype rows; cast explicitly
        topic_colors = {
            f"topic_{int(row['topic'])}": cmap(0.15 + 0.7 * (row['mean_pagerank'] - pr_min) / pr_range)
            for _, row in df_sorted.iterrows()
        }
        paper_colors = {
            f"paper_{pid}": topic_colors[f"topic_{tid}"]
            for tid, papers in topic_to_papers.items()
            for pid in papers
        }

        # --- Node sizes ---
        hub_size_scale = 3000
        hub_sizes = [
            300 + hub_size_scale * (G.nodes[n]['pagerank'] - pr_min) / pr_range
            for n in topic_node_ids
        ]
        paper_node_ids = [n for n in G.nodes if G.nodes[n]['node_type'] == 'paper']
        paper_sizes = [30] * len(paper_node_ids)

        # --- Draw ---
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.set_facecolor('#f8f8f8')
        fig.patch.set_facecolor('#f8f8f8')

        # Edges (spokes) — thin and low-alpha
        if paper_node_ids:
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12,
                                   edge_color='#888888', width=0.5)

        # Paper satellite nodes
        if paper_node_ids:
            paper_node_color_list = [paper_colors[n] for n in paper_node_ids]
            nx.draw_networkx_nodes(G, pos, nodelist=paper_node_ids,
                                   node_size=paper_sizes,
                                   node_color=paper_node_color_list,
                                   alpha=0.45, ax=ax)

        # Topic hub nodes
        hub_color_list = [topic_colors[n] for n in topic_node_ids]
        nx.draw_networkx_nodes(G, pos, nodelist=topic_node_ids,
                               node_size=hub_sizes,
                               node_color=hub_color_list,
                               alpha=0.92, ax=ax,
                               edgecolors='white', linewidths=1.5)

        # Topic labels — offset slightly above hub centre
        label_map = {f"topic_{t}": self.get_topic_label(t) for t in topic_ids}
        label_pos = {k: (v[0], v[1] + 0.04) for k, v in hub_pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=label_map,
                                font_size=7, font_weight='bold',
                                font_color='#222222', ax=ax)

        # Colorbar for PageRank
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=mcolors.Normalize(vmin=pr_min, vmax=pr_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
        cbar.set_label('Mean PageRank', fontsize=10)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
                   markersize=12, label='Topic (size = PageRank)'),
        ]
        if paper_node_ids:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
                       markersize=5, alpha=0.5, label='Individual paper')
            )
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
                  framealpha=0.8)

        ax.set_title('Topic PageRank Centrality Network', fontsize=15, fontweight='bold', pad=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Centrality PageRank network saved to {output_file}")

    def visualize_influence_citation_activity(self, df, output_file='influence_citation_activity.png',
                                              normalize_by_works=False):
        """
        Grouped bar chart: average citations received vs given per topic.

        Topics are sorted by citations received (descending).

        Parameters
        ----------
        df : pd.DataFrame
            Influence data.
        output_file : str
            Path to save the PNG.
        normalize_by_works : bool
            If True, divide citation counts by n_works before plotting.
        """
        df = df.sort_values('avg_citations_received', ascending=False).reset_index(drop=True)
        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        received = df['avg_citations_received'].copy()
        given = df['avg_citations_given'].copy()

        if normalize_by_works and 'n_works' in df.columns:
            received = received / df['n_works'].replace(0, np.nan)
            given = given / df['n_works'].replace(0, np.nan)
            y_label = 'Citations per Work'
            title_suffix = ' (per work)'
        else:
            y_label = 'Average Citations'
            title_suffix = ''

        fig_h = max(6, len(topic_labels) * 0.55)
        fig, ax = plt.subplots(figsize=(12, fig_h))

        # Sort ascending so top topic appears at the top of the chart
        y_pos = np.arange(len(topic_labels))
        height = 0.38
        ax.barh(y_pos + height / 2, received, height,
                label='Received', alpha=0.85, color='#d0d1e6')
        ax.barh(y_pos - height / 2, given, height,
                label='Given', alpha=0.85, color='#67a9cf')

        import textwrap
        wrapped_labels = [textwrap.fill(lbl, width=35) for lbl in topic_labels]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels[::-1], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(y_label, fontsize=7)
        ax.set_title(f'Citation Activity by Topic{title_suffix}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(left=0.35)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Citation activity chart saved to {output_file}")

    def visualize_influence_internal_vs_external(self, df,
                                                  output_file='influence_internal_vs_external.png',
                                                  normalize_by_works=False):
        """
        Scatter plot: internal vs external citations received per topic.

        Quadrant lines are drawn at the median of each axis.

        Parameters
        ----------
        df : pd.DataFrame
            Influence data.
        output_file : str
            Path to save the PNG.
        normalize_by_works : bool
            If True, divide citation counts by n_works before plotting.
        """
        internal = df['total_internal_citations_received'].copy()
        external = df['total_external_citations'].copy()

        if normalize_by_works and 'n_works' in df.columns:
            n_works = df['n_works'].replace(0, np.nan)
            internal = internal / n_works
            external = external / n_works
            x_label = 'Internal Citations Received (per work)'
            y_label = 'External Citations Received (per work)'
        else:
            x_label = 'Internal Citations Received'
            y_label = 'External Citations Received'

        topic_labels = [self.get_topic_label(t) for t in df['topic']]
        sizes = df['n_works'] * 0.6 if 'n_works' in df.columns else 60

        fig, ax = plt.subplots(figsize=(13, 10))

        scatter = ax.scatter(internal, external, s=sizes,
                             alpha=0.65, color='#014636',
                             edgecolors='white', linewidths=0.6)

        x_med = internal.median()
        y_med = external.median()

        # Quadrant labels in corners
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        quadrant_style = dict(fontsize=7.5, color='#555555', alpha=0.75,
                              ha='center', va='center')
        ax.text(x_med * 0.5, y_max * 0.96, 'Low Internal\nHigh External', **quadrant_style)
        ax.text(x_max * 0.88, y_max * 0.96, 'High Internal\nHigh External', **quadrant_style)
        ax.text(x_med * 0.5, y_med * 0.25, 'Low Internal\nLow External', **quadrant_style)
        ax.text(x_max * 0.88, y_med * 0.25, 'High Internal\nLow External', **quadrant_style)

        # Label only top-20 topics by total citations to avoid overcrowding
        total_citations = internal + external
        label_mask = total_citations >= total_citations.nlargest(min(20, len(df))).min()
        texts = []
        for i in df[label_mask.values].index:
            idx = list(df.index).index(i)
            texts.append(ax.text(internal.iloc[idx], external.iloc[idx],
                                 topic_labels[idx], fontsize=6.5, alpha=0.8))
        adjust_text(texts, ax=ax,
                    force_text=(0.3, 0.5),
                    force_points=(0.6, 0.12),
                    expand=(1.2, 1.5),
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.2, shrinkA=5))

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title('Internal vs External Citations Received by Topic\n'
                     '(point size = n_works; dashed lines = medians)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Internal vs external citations chart saved to {output_file}")

    def visualize_trends_growth(self, df, output_file='trends_growth.png',
                                color_growing='#014636', color_declining='#d0d1e6'):
        """
        Horizontal bar chart of topic trend slopes, sorted descending.

        Parameters
        ----------
        df : pd.DataFrame
            Trends data with 'topic' and 'trend_slope' columns.
        output_file : str
            Path to save the PNG.
        color_growing : str
            Hex/named colour for positive (growing) bars.
        color_declining : str
            Hex/named colour for negative (declining) bars.
        """
        df = df.sort_values('trend_slope', ascending=True).reset_index(drop=True)
        topic_labels = [self.get_topic_label(t) for t in df['topic']]
        colors = [color_growing if v > 0 else color_declining for v in df['trend_slope']]

        fig_h = max(6, len(topic_labels) * 0.4)
        fig, ax = plt.subplots(figsize=(13, fig_h))

        import textwrap
        wrapped_labels = [textwrap.fill(lbl, width=35) for lbl in topic_labels]
        ax.barh(wrapped_labels, df['trend_slope'], color=colors, alpha=0.82)
        ax.axvline(x=0, color='#333333', linewidth=1.0)
        ax.set_xlabel('Trend Slope (publications per year)', fontsize=11)
        ax.set_title('Topic Growth Trends',
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(left=0.35)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Growth trends chart saved to {output_file}")

    def visualize_trends_temporal_spans(self, df, output_file='trends_temporal_spans.png',
                                        top_n=25, sort_by='total_works',
                                        color_growing='#014636', color_declining='#d0d1e6'):
        """
        Gantt-style chart of topic temporal spans, coloured by trend direction.

        Parameters
        ----------
        df : pd.DataFrame
            Trends data with 'topic', 'first_year', 'last_year', 'trend_slope',
            and optionally 'total_works' columns.
        output_file : str
            Path to save the PNG.
        top_n : int
            Maximum number of topics to show (filtered by sort_by metric).
        sort_by : str
            Column to rank topics by before taking top_n.
            Options: 'total_works', 'first_year', 'span_length', 'trend_slope'.
        color_growing : str
            Hex/named colour for topics with positive trend slope.
        color_declining : str
            Hex/named colour for topics with negative trend slope.
        """
        df = df.copy()
        df['span_length'] = df['last_year'] - df['first_year']

        # Determine sort column, fall back to total_works or span_length
        if sort_by not in df.columns:
            sort_by = 'span_length'
        df = df.nlargest(top_n, sort_by).sort_values('first_year').reset_index(drop=True)

        topic_labels = [self.get_topic_label(t) for t in df['topic']]

        # Colour lines by span length (longest = darkest) using a single sequential colormap
        span_vals = df['span_length'].values.astype(float)
        span_min, span_max = span_vals.min(), span_vals.max()
        span_range = span_max - span_min if span_max > span_min else 1.0
        cmap = plt.cm.Blues
        line_colors = [cmap(0.35 + 0.6 * (s - span_min) / span_range) for s in span_vals]

        fig_h = max(6, len(df) * 0.35)
        fig, ax = plt.subplots(figsize=(14, fig_h))

        for i, (_, row) in enumerate(df.iterrows()):
            ax.plot(
                [row['first_year'], row['last_year']], [i, i],
                color=line_colors[i], linewidth=2.5, solid_capstyle='round'
            )
            # Endpoint dots
            ax.scatter([row['first_year'], row['last_year']], [i, i],
                       color=line_colors[i], s=18, zorder=3)
            # Year annotations with a small offset so text doesn't touch the dot
            year_buf = (df['last_year'].max() - df['first_year'].min()) * 0.01
            ax.text(row['first_year'] - year_buf, i, str(int(row['first_year'])),
                    ha='right', va='center', fontsize=7, color='#444444')
            ax.text(row['last_year'] + year_buf, i, str(int(row['last_year'])),
                    ha='left', va='center', fontsize=7, color='#444444')

        import textwrap
        wrapped_labels = [textwrap.fill(lbl, width=35) for lbl in topic_labels]
        ax.set_yticks(range(len(wrapped_labels)))
        ax.set_yticklabels(wrapped_labels, fontsize=8)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_title(f'Topic Temporal Spans (top {len(df)} by {sort_by})',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Colorbar to indicate span length
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=int(span_min), vmax=int(span_max))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.015, pad=0.01)
        cbar.set_label('Span length (years)', fontsize=9)

        plt.tight_layout()
        plt.subplots_adjust(left=0.35)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temporal spans chart saved to {output_file}")

    def visualize_geo(self, metadata_df, output_file='geo_publications_map.png',
                      country_col='authorships.countries',
                      geo_cmap='YlOrRd', geo_format='static'):
        """
        Choropleth map of publication geographic distribution.

        Counts are computed from the FULL metadata DataFrame (entire corpus,
        no outlier filtering) because geography is corpus-level, not topic-level.

        A paper with multiple country affiliations (pipe-separated) contributes
        one count to each unique country it lists.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Full metadata CSV loaded as a DataFrame. Must contain `country_col`.
        output_file : str
            Path to save the output file (.png for static, .html for interactive).
        country_col : str
            Column name containing pipe-separated ISO 3166 alpha-2 country codes.
        geo_cmap : str
            Matplotlib colormap name (static) or Plotly color scale name (interactive).
        geo_format : str
            'static' saves a PNG via matplotlib/geopandas.
            'interactive' saves an HTML via plotly.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for geographic visualization.\n"
                "Install with: pip install geopandas geodatasets"
            )
        try:
            import pycountry
        except ImportError:
            raise ImportError("pycountry is required. Install with: pip install pycountry")

        if country_col not in metadata_df.columns:
            raise ValueError(
                f"Column '{country_col}' not found in metadata.\n"
                f"Available columns: {list(metadata_df.columns)}"
            )

        total_papers = len(metadata_df)
        print(f"  Building country counts from {total_papers} records...")

        # --- Parse and count ---
        country_series = (
            metadata_df[country_col]
            .dropna()
            .astype(str)
            .str.split('|')
            .explode()
            .str.strip()
            .str.upper()
        )
        # Drop empty strings and obvious non-codes
        country_series = country_series[country_series.str.len() == 2]
        country_counts = country_series.value_counts()

        # alpha-2 → alpha-3 using pycountry
        def _to_alpha3(code):
            try:
                return pycountry.countries.get(alpha_2=code).alpha_3
            except AttributeError:
                return None

        geo_rows = []
        unmapped = []
        for alpha2, count in country_counts.items():
            alpha3 = _to_alpha3(alpha2)
            if alpha3:
                geo_rows.append({
                    'iso_a3': alpha3,
                    'alpha_2': alpha2,
                    'n_publications': count,
                    'proportion': count / total_papers * 100,
                })
            else:
                unmapped.append(alpha2)

        if unmapped:
            print(f"  ⚠ Could not map {len(unmapped)} country codes to alpha-3: {unmapped}")

        geo_df = pd.DataFrame(geo_rows)
        print(f"  Mapped {len(geo_df)} countries; top 5:")
        for _, r in geo_df.nlargest(5, 'proportion').iterrows():
            print(f"    {r['alpha_2']}  {r['proportion']:.2f}%  (n={int(r['n_publications'])})")

        # --- Load world geometries (handle geopandas < 1.0 and >= 1.0) ---
        world = None
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except AttributeError:
            # geopandas >= 1.0 removed built-in datasets; geodatasets no longer ships
            # naturalearth.lowres, so fetch the 110m countries shapefile directly.
            _NE_COUNTRIES_URL = (
                "https://naciscdn.org/naturalearth/110m/cultural/"
                "ne_110m_admin_0_countries.zip"
            )
            try:
                import pooch
                _cache_path = pooch.retrieve(
                    url=_NE_COUNTRIES_URL,
                    known_hash=None,
                    fname="ne_110m_admin_0_countries.zip",
                )
                world = gpd.read_file(_cache_path)
            except Exception:
                # Last resort: read directly from URL (requires network)
                try:
                    world = gpd.read_file(_NE_COUNTRIES_URL)
                except Exception:
                    pass

        if world is None:
            raise RuntimeError(
                "Could not load Natural Earth world geometries.\n"
                "Ensure you have network access, or install pooch: pip install pooch"
            )

        # Normalize ISO A3 column name (lowres uses 'iso_a3'; full dataset uses 'ISO_A3')
        if 'iso_a3' not in world.columns and 'ISO_A3' in world.columns:
            world = world.rename(columns={'ISO_A3': 'iso_a3'})

        # drop rows where iso_a3 is '-99' (unmapped territories)
        world = world[world['iso_a3'] != '-99'].copy()

        merged = world.merge(geo_df, on='iso_a3', how='left')
        merged['proportion'] = merged['proportion'].fillna(0)

        # --- Plot ---
        if geo_format == 'interactive':
            import plotly.express as px

            # Ensure output file has .html extension
            if not output_file.endswith('.html'):
                output_file = output_file.rsplit('.', 1)[0] + '.html'

            fig = px.choropleth(
                geo_df,
                locations='iso_a3',
                color='proportion',
                hover_name='alpha_2',
                hover_data={'proportion': ':.2f%', 'n_publications': True},
                color_continuous_scale=geo_cmap,
                labels={'proportion': '% of corpus', 'n_publications': 'Publications'},
                title='Geographic Distribution of Corpus Publications (% of total)',
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True),
                margin=dict(l=0, r=0, t=50, b=0),
            )
            fig.write_html(output_file)
            print(f"Interactive geo map saved to {output_file}")

        else:
            # Ensure output file has .png extension
            if not output_file.endswith('.png'):
                output_file = output_file.rsplit('.', 1)[0] + '.png'

            fig, ax = plt.subplots(figsize=(18, 10))
            ax.set_facecolor('#d0e8f0')
            fig.patch.set_facecolor('white')

            # Countries with no data in a neutral grey
            merged.plot(ax=ax, color='#d3d3d3', linewidth=0.3, edgecolor='white')

            # Countries with data coloured by proportion
            merged[merged['proportion'] > 0].plot(
                column='proportion',
                cmap=geo_cmap,
                linewidth=0.3,
                edgecolor='white',
                legend=True,
                legend_kwds={
                    'label': '% of corpus publications',
                    'shrink': 0.45,
                    'orientation': 'horizontal',
                    'pad': 0.02,
                },
                ax=ax,
            )

            ax.set_title(
                'Geographic Distribution of Corpus Publications\n'
                f'(proportion of total {total_papers:,} records; '
                f'grey = no affiliation data)',
                fontsize=13, fontweight='bold', pad=12,
            )
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Static geo map saved to {output_file}")

    def generate_all_visualizations(self,
                                   centrality_file='centrality.csv',
                                   influence_file='influence.csv',
                                   trends_file='trends.csv',
                                   works_file=None,
                                   normalize_by_works=False,
                                   top_n_spans=25,
                                   sort_spans_by='total_works',
                                   color_growing='#2ecc71',
                                   color_declining='#e74c3c',
                                   metadata_df=None,
                                   geo_cmap='YlOrRd',
                                   geo_format='static'):
        """Generate all visualizations at once"""
        print("Starting bibliographic analysis visualizations...")

        try:
            df_centrality = self.load_data(centrality_file)
            self.visualize_centrality_pagerank_network(df_centrality, works_file=works_file)
        except FileNotFoundError:
            print(f"Warning: {centrality_file} not found, skipping centrality visualization")

        try:
            df_influence = self.load_data(influence_file)
            self.visualize_influence_citation_activity(df_influence,
                                                       normalize_by_works=normalize_by_works)
            self.visualize_influence_internal_vs_external(df_influence,
                                                          normalize_by_works=normalize_by_works)
        except FileNotFoundError:
            print(f"Warning: {influence_file} not found, skipping influence visualization")

        try:
            df_trends = self.load_data(trends_file)
            self.visualize_trends_growth(df_trends,
                                         color_growing=color_growing,
                                         color_declining=color_declining)
            self.visualize_trends_temporal_spans(df_trends,
                                                  top_n=top_n_spans,
                                                  sort_by=sort_spans_by,
                                                  color_growing=color_growing,
                                                  color_declining=color_declining)
        except FileNotFoundError:
            print(f"Warning: {trends_file} not found, skipping trends visualization")

        if metadata_df is not None:
            ext = 'html' if geo_format == 'interactive' else 'png'
            self.visualize_geo(metadata_df,
                               output_file=f'geo_publications_map.{ext}',
                               geo_cmap=geo_cmap,
                               geo_format=geo_format)

        print("\nAll visualizations complete!")