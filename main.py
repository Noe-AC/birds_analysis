"""
Résumé :
- Détection d'oiseaux via BirdNet et analyse des détections.

Crédit :
- Noé Aubin-Cadot

Dépendances :
- pip install birdnetlib
- pip install librosa
- pip install resampy
- pip install adjustText

--------------------------------------------------------

BirdNet Analyser :

Source :
	https://github.com/kahst/BirdNET-Analyzer?tab=readme-ov-file

Installation :
	pip install birdnetlib
	pip install librosa
	pip install resampy

--------------------------------------------------------

"""


################################################################################
################################################################################
# Importer des librairies

import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta

################################################################################
################################################################################
# Définition de fonctions

# Définition d'une fonction qui prend un fichier d'entrée et rend la date correspondante
def input_file_to_date(
	input_file,
):
	date_str = input_file.split('/')[-1].split('.')[0].split(' ')[0]
	return date_str

# Définition d'une fonction qui prend une liste de fichiers audio et rend un DataFrame des oiseaux détectés
def input_files_to_DataFrame(
	input_files,
	output_file,
	latitude  = 45.50884,  # Latitude de Montréal
	longitude = -73.58781, # Longitude de Montréal
	min_conf  = 0.05,      # Niveau de confiance minimal
	date      = None,      # Si on ne veut analyser qu'une date spécifique (format string 'YYYY-MM-DD')
	verbose   = False,     # Verbosité
):
	# Librairies pour BirdNet
	from birdnetlib import Recording
	from birdnetlib.analyzer import Analyzer

	# Si une date est spécifiée, on filtre les fichiers à analyser
	if date:
		input_files_to_analyse = [input_file for input_file in input_files if input_file_to_date(input_file)==date]

	# On déclare une liste dans laquelle on va mettre les DataFrames des détections d'oiseaux
	list_of_df = []

	# Un compteur pour spécifier où la boucle est rendue
	counter = 0

	# On fait une boucle sur les fichiers audio
	for input_file in sorted(input_files_to_analyse):
		
		# On affiche où on est rendu
		counter += 1
		print(f'\nFichier audio {counter}/{len(input_files_to_analyse)}.')
		print('input_file =',input_file)

		# On crée un objet datetime
		datetime_str = input_file.split('/')[-1].split('.')[0]
		datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H%M")

		# Détection des oiseaux via BirdNet
		recording = Recording(
			analyzer = Analyzer(),
			path     = input_file,
			lat      = latitude,
			lon      = longitude,
			date     = datetime_obj,
			min_conf = min_conf,
		)
		recording.analyze()

		# On prend la liste des nouvelles détections
		l_detections = recording.detections

		# On affiche le nombre de détections pour ce fichier audio
		print('Nombre de détections :',len(l_detections))

		# Conversion de la liste de détections en DataFrame
		df_detections = pd.DataFrame(l_detections)

		# On ajoute une colonne qui spécifie la durée de la détection
		df_detections['duration'] = df_detections['end_time'] - df_detections['start_time']

		# On ajoute une colonne qui spécifie la date et le datetime du fichier
		df_detections['file_date']     = datetime_obj.strftime('%Y-%m-%d')
		df_detections['file_datetime'] = datetime_obj.strftime('%Y-%m-%d %H:%M')

		# On ajoute une colonne qui spécifie le datetime de début et de fin de détection de l'oiseau
		df_detections['bird_datetime_start'] = df_detections['start_time'].apply(lambda x:(datetime_obj+timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M'))
		df_detections['bird_datetime_end']   = df_detections['end_time'].apply(lambda x:(datetime_obj+timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M'))       

		# On ajoute les oiseaux détectés à la liste de détections d'oiseaux
		list_of_df.append(df_detections)

	# Génération d'un DataFrame qui contient toutes les détections
	df_detections = pd.concat(
		objs = list_of_df,
		axis = 0,
		ignore_index = True,
	)

	# On réordonne les colonnes
	cols = [
		'file_datetime',
		'file_date',
		'bird_datetime_start',
		'bird_datetime_end',
		'start_time',
		'end_time',
		'duration',
		'confidence',
		'common_name',
		'scientific_name',
		'label',
	]
	df_detections = df_detections[cols]

	if verbose:
		print('\ndf_detections.shape =',df_detections.shape)
		print('\ndf_detections :\n',df_detections,sep='')

	# On exporte les résultats
	df_detections.to_csv(output_file,index=False)

def translate_birds(
	input_file_detections,
	input_file_traductions,
	output_file,
):
	# On importe les données des détections
	print('Importing :',input_file_detections)
	df_detections  = pd.read_csv(input_file_detections)

	# On importe les données des traductions
	print('Importing :',input_file_traductions)
	df_traductions = pd.read_csv(input_file_traductions)

	# On regarde les oiseaux détectés qui ne sont pas dans les oiseaux dont la traduction est connue
	set_detected   = set(df_detections['common_name'].sort_values().drop_duplicates())
	set_translated = set(df_traductions['oiseau_en'].sort_values())
	birds_not_translated = sorted(set_detected-set_translated)
	if len(birds_not_translated)>0:
		print('Oiseaux non traduits :')
		for bird in birds_not_translated:
			print(bird)
		raise Exception('Translations missing... please fill in the translation file for the birds mentioned above.')

	# On traduit par une jointure
	df_traductions.rename(columns={'oiseau_en':'common_name'},inplace=True)

	# On fait une jointure
	df_detections = pd.merge(
		left  = df_detections,
		right = df_traductions,
		on    = 'common_name',
		how   = 'left',
	)

	# On exporte
	print('Exporting :',output_file)
	df_detections.to_csv(output_file,index=False)
	print('Done.')

def make_path(path,verbose=False):
	import os
	path_components = path.split('/')
	if '' in path_components:
		path_components.remove('')
	for i in range(1,len(path_components)+1):
		folder = '/'.join(path_components[:i])
		if not os.path.exists(folder):
			if verbose:
				print(f"Folder '{folder}' does not exist, it is created.")
			os.mkdir(folder)
		else:
			if verbose:
				print(f"Folder '{folder}' already exists, continue.")

def export_figure(
	plt,
	output_file_fig = None,
	do_make_path    = False,
	do_clean_memory = False,
	verbose         = True
):
	if output_file_fig==None:
		plt.show()
	else:
		if do_make_path:
			output_path = '/'.join(output_file_fig.split('/')[:-1])
			if verbose:
				print('Making path :',output_path)
			make_path(output_path)
		if verbose:
			print('Exporting :',output_file_fig)
		plt.savefig(output_file_fig,dpi=300)
		plt.close()
		if do_clean_memory:
			plt.cla()
			plt.clf()
			plt.close('all')
			plt.close(fig)
			import gc
			gc.collect()
		if verbose:
			print('Done.')


def show_barplot_confidence(
	df,
	output_file_fig = None,
):
	df = df.copy() # Pour éviter modifications in-place
	df.sort_values(by=['oiseau_fr','confidence'],ascending=[True,False],inplace=True)
	df.drop_duplicates(subset=['oiseau_fr'],inplace=True)
	df.sort_values(by='confidence',ascending=False,inplace=True)
	df = df[['oiseau_fr','confidence']]
	print(df.to_string(index=False))
	import matplotlib.pyplot as plt
	ax = df.head(50).sort_values(by='confidence',ascending=True).plot(
		kind    = 'barh',
		x       = 'oiseau_fr',
		y       = 'confidence',
		figsize = (12,8),
	)
	ax.tick_params(axis='y', labelsize=11)
	plt.title('Confiance maximale détectée par espèce (confiance ≥ 0.25)',size=18)
	plt.xlabel('Confiance maximale détectée',size=15)
	plt.ylabel("Espèce d'oiseau",size=15)
	xticks = np.arange(0.0,1.05,0.05).round(2)
	plt.xticks(xticks,[format(x,'0.2f') for x in xticks])
	plt.xlim(0,1)
	plt.grid()
	plt.tight_layout()
	export_figure(
		plt             = plt,
		output_file_fig = output_file_fig,
		do_make_path    = True,
	)

def show_barplot_frequence(
	df,
	output_file_fig = None,
):
	df = df.copy() # Pour éviter modifications in-place
	s_count = df['oiseau_fr'].value_counts().sort_values(ascending=True)
	print(s_count.sort_values(ascending=False))
	import matplotlib.pyplot as plt
	ax = s_count.plot(
		kind    = 'barh',
		figsize = (12,8),
	)
	ax.tick_params(axis='y', labelsize=11)
	plt.title('Nombre de détections (confiance ≥ 0.25)',size=18)
	plt.xlabel('Nombre de détections',size=15)
	plt.ylabel("Espèce d'oiseau",size=15)
	plt.xticks(np.arange(0,95,5),np.arange(0,95,5))
	plt.grid()
	plt.tight_layout()
	export_figure(
		plt             = plt,
		output_file_fig = output_file_fig,
		do_make_path    = True,
	)

def show_scatterplot_freq_conf(
	df,
	output_file_fig = None,
):
	# Pour chaque oiseau on calcule le nombre de fois qu'il a été détecté
	s_count = df['oiseau_fr'].value_counts().sort_index()
	s_count.name = 'count'
	# Pour calculer la confiance moyenne sur les 5 meilleures détections par oiseau
	df_sorted  = df.sort_values(by=['oiseau_fr', 'confidence'], ascending=[True, False])
	df_top     = df_sorted[['oiseau_fr','confidence']].groupby('oiseau_fr').head(5)
	s_top      = df_top.groupby('oiseau_fr')['confidence'].mean().sort_index()
	s_top.name = 'confidence_mean'
	# Concaténation du dénombrement et des confiances moyennes des 5 plus confiantes observations
	df_count_top = pd.concat((s_count,s_top),axis=1)
	print(df_count_top)
	import matplotlib.pyplot as plt
	df_count_top.plot(
		kind    = 'scatter',
		x       = 'count',
		y       = 'confidence_mean',
		figsize = (12,8),
		s       = 5,
	)
	plt.title('Confiance moyenne vs nombre de détections (confiance ≥ 0.25)',size=18)
	plt.xlabel('Nombre de détections',size=15)
	plt.ylabel('Moyenne de la confiance des 5 plus confiantes détections',size=15)
	plt.xticks(np.arange(0,100,5),np.arange(0,100,5))
	yticks = np.arange(0.2,1.05,0.05).round(2)
	plt.yticks(yticks,[format(y,'0.2f') for y in yticks])
	# https://adjusttext.readthedocs.io/en/latest/_modules/adjustText.html
	bbox = {
		'facecolor' : 'red',
		'alpha'     : 0.05,
	}
	texts = [plt.text(x=x,y=y,s=bird, bbox=bbox) for x, y, bird in zip(df_count_top['count'], df_count_top['confidence_mean'], df_count_top.index.to_list())]
	from adjustText import adjust_text # pip install adjustText
	adjust_text(
		texts           = texts,
		arrowprops      = dict(arrowstyle="->", color='r', lw=0.5, mutation_scale=10),
		connectionstyle = 'arc3,rad=0.3',
	)
	plt.grid()
	plt.tight_layout()
	export_figure(
		plt             = plt,
		output_file_fig = output_file_fig,
		do_make_path    = True,
	)

def analyse_birds(
	input_file,
	output_file_fig_confidence = None,
	output_file_fig_frequence  = None,
	output_file_fig_freq_conf  = None,
	datetime_start             = None,
	datetime_end               = None,
):
	# On importe les données
	print('Importing :',input_file)
	df = pd.read_csv(input_file)

	# Si on veut filtrer sur la plage horaire des observations
	if datetime_start:
		df = df[df['file_datetime']>=datetime_start]
	if datetime_end:
		df = df[df['file_datetime']<=datetime_end]

	# On ne garde que les détections qui ont au moins 0.25 de confiance
	df = df[df['confidence']>=0.25]


	do_show_barplot_confidence=1
	if do_show_barplot_confidence:
		show_barplot_confidence(
			df              = df,
			output_file_fig = output_file_fig_confidence,
		)
		

	do_show_barplot_frequence=1
	if do_show_barplot_frequence:
		show_barplot_frequence(
			df              = df,
			output_file_fig = output_file_fig_frequence,
		)

	do_show_scatterplot_freq_conf=1
	if do_show_scatterplot_freq_conf:
		show_scatterplot_freq_conf(
			df              = df,
			output_file_fig = output_file_fig_freq_conf,
		)



################################################################################
################################################################################
# Utilisation de fonctions

def main():

	date = '2024-05-16' # Date où les fichiers audio ont été enregistrés

	# Une fonction pour créer un fichier CSV d'oiseaux détectés à partir de fichiers audio
	do_input_files_to_DataFrame=1
	if do_input_files_to_DataFrame:
		input_files_to_DataFrame(
			input_files = sorted(glob.glob(f'input/audio/{date} *.wav')), # En entrée on prend les fichiers audio WAV
			output_file = f'output/{date}_df_detections_local_en.csv',    # En sortie on obtient un fichier CSV de détections d'oiseaux
			date        = date,
			verbose     = True,
		)

	# Une fonction pour traduire les oiseaux de l'anglais vers le français
	do_translate_birds=1
	if do_translate_birds:
		translate_birds(
			input_file_detections  = f'output/{date}_df_detections_local_en.csv',
			input_file_traductions = 'traduction/translation_birds_french_to_english.csv',
			output_file            = f'output/{date}_df_detections_local_fr.csv',
		)

	# Une fonction pour anlayser les détections d'oiseaux
	do_analyse_birds=1
	if do_analyse_birds:
		analyse_birds(
			input_file                 = f'output/{date}_df_detections_local_fr.csv',
			output_file_fig_confidence = f'figures/{date}/confidence.png',
			output_file_fig_frequence  = f'figures/{date}/frequence.png',
			output_file_fig_freq_conf  = f'figures/{date}/freq_conf.png',
		)

if __name__ == '__main__':
	main()









