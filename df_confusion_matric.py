import pandas as pd 
from pandas import ExcelWriter

slidewise_threshold = {
    'BACTERIA':{'lower':0.5, 'higher':1},
'YEAST':{'lower':0.5, 'higher':1},
'FUNGUS':{'lower':0.4, 'higher':1}
}

# slidewise_threshold = {
#     'BACTERIA':{'lower':0.5, 'higher':0.6},
# 'YEAST':{'lower':0.5, 'higher':0.6},
# 'FUNGUS':{'lower':0.5, 'higher':0.6}
# }

stdh = slidewise_threshold

def check_pathogens(pred_data,data):
    # print(data)
    # df3 = pd.merge(data,pred_data , on=['image_name'], how='left')
	# print('pred_data', pred_data.columns)
	# print(data.head())
    df3 = pd.merge(data,pred_data, on=['image_name'])
    # df3 = df3.loc[:, ~data.columns.str.contains('^Unnamed')]
    # print("KEYS", df3.columns)
    # KEYS Index(['Unnamed: 0', 'image_name', 'PATHOGENS_x', 'class_id', 'score',
    #    'PATHOGENS_y'],
    #   dtype='object')
    # df3["score"].astype(float)
    df3=df3[['image_name','PATHOGENS_x','PATHOGENS_y' ,'score' ]]
    # df3=df3[['image_name','PATHOGENS_x','PATHOGENS_y' ,'score' ]]
    # print(df3)
    return df3 

def creation_of_confusion_matric(data,final_out_csv,stain=None):
    # print('stain', stain)
	d={}
	confunsion_dict={}
	confusion_metric_list=['TP','FP','FN','TN']
	confusion_metric_dict={}
	for col in data.columns: 
		if col!='image_name':
			val=list(set(data[col]))
			d.update({col:val})
	# print(d)

	for i in confusion_metric_list:
		confusion_metric_dict.update({i:{}})
			
	for key in confusion_metric_dict:
		# print(key)
		# print('key',key)
		if key=='TP':
			met={}
			for key1 in d:
				if '_x' in key1 :   
					met.update({key1:[stain]})
				elif '_y' in key1:
					met.update({key1:[stain]})
				print('met',key,met)
				confusion_metric_dict.update({key:met})        

		elif key=='FP':
			met={}
			for key1 in d:
				if '_x' in key1:
					val=d[key1]
					if stain not in val:
						met.update({key1:val}) 
					else:
						val.remove(stain)
						met.update({key1:val}) 
				elif '_y' in key1:
						met.update({key1:[stain]})
				confusion_metric_dict.update({key:met})  
		elif key=='FN':
			met={}
			for key1 in d:
				if '_x' in key1:
						met.update({key1:[stain]})    
				elif '_y' in key1:
					val=d[key1]
					if stain not in val:
						met.update({key1:val}) 
					else:
						val.remove(stain)
						met.update({key1:val})      
				confusion_metric_dict.update({key:met})
			# print('confusion_metric_dict', confusion_metric_dict)
		elif key=='TN':
			met={}
			for key1 in d:
				if '_x' in key1:
					val=d[key1]
						# val.remove(stain)
					met.update({key1:val})
				elif '_y' in key1:
					val=d[key1]
						# val.remove(stain)
					met.update({key1:val})
				confusion_metric_dict.update({key:met})    
	# print(confusion_metric_dict) == 
	# {'TP': {'PATHOGENS_x': ['YEAST'], 'PATHOGENS_y': ['YEAST']}, 
	# 'FP': {'PATHOGENS_x': [nan, 'YEAST,JUNK', 'NEGATIVE', 'JUNK'], 'PATHOGENS_y': ['YEAST']}, 
	# 'TN': {'PATHOGENS_x': [nan, 'YEAST,JUNK', 'NEGATIVE', 'JUNK'], 'PATHOGENS_y': [nan, 'FUNGUS', 'BACTERIA', 'JUNK', 'EMPTY']}, 
	# 'FN': {'PATHOGENS_x': ['YEAST'], 'PATHOGENS_y': [nan, 'FUNGUS', 'BACTERIA', 'JUNK', 'EMPTY']}}

	for key2 in confusion_metric_dict:
		print('key2', key2)
		value=confusion_metric_dict[key2]
		print('jjj',value)
		val=value['PATHOGENS_x']
		# print('PATHOGENS_x', val)
		val1=value['PATHOGENS_y']
		# print('PATHOGENS_y', val1)
		# rslt_df = data[(data['PATHOGENS_x'].isin(val))& (data['PATHOGENS_y'].isin(val1)) & data['score'].between(0.6, 0.9)]
		# # print(rslt_df)
		# print("w/o filter ", key2 ,rslt_df.shape[0])
		# # rslt_df.dropna(inplace=True)
		# rslt_df = rslt_df.dropna(subset=['image_name'])
		# rslt_df = rslt_df['image_name'].unique()l
		# print("w/ filter ", key2 ,rslt_df.shape[0])
		# # print(rslt_df)
		# confunsion_dict.update({key2:rslt_df.shape[0]})
		# print(data.head())
		rslt_df = data[(data['PATHOGENS_x'].isin(val))& (data['PATHOGENS_y'].isin(val1))]
		# rslt_df = data[(data['PATHOGENS_x'].isin(val))& (data['PATHOGENS_y'].isin(val1)) & data['score'].between(stdh[stain]['lower'], stdh[stain]['higher'])]
		# rslt_df.to_csv('/home/kalpit/Desktop/zero_files/cleaned/{0}_orignal.csv'.format(key2))
		# print(rslt_df)
		# print("w/o filter ", key2 ,rslt_df.shape[0])
		# rslt_df.dropna(inplace=True)
		# rslt_df = rslt_df.dropna(subset=['image_name'])
		# print('type11',rslt_df.head())
		# rslt_df.to_csv('/home/kalpit/Desktop/zero_files/cleaned/{0}_wof.csv'.format(key2))
		# print(rslt_df.columns)
		# rslt_df_new = rslt_df.drop_duplicates(subset=['image_name'], keep='last')
		# print('type22',type(rslt_df))
		# rslt_df_new.to_csv('/home/kalpit/Desktop/zero_files/cleaned/{0}_wfpas.csv'.format(key2))
		# df_u = rslt_df.sort_values('score').drop_duplicates(subset=['image_name', 'PATHOGENS_x','PATHOGENS_y'], keep='last')
		# print('df_u',df_u.head(2))
		print('final_out_csv',final_out_csv,key2)
		# # writer_e = pd.ExcelWriter(final_out_csv, engine='xlsxwriter')
		# if not rslt_df.empty:
		# with ExcelWriter(final_out_csv, mode='a',engine='openpyxl') as writer:
		# 	print('writer',key2)
		# 	rslt_df.to_excel(writer, sheet_name=key2)
				# writer.save()
		
		# df_u.to_excel(writer_e, sheet_name=key2)
		# writer_e.save()
		rslt_df = rslt_df['image_name'].unique()
		# rslt_df.to_csv('/home/kalpit/Desktop/zero_files/cleaned/{0}_orignal.csv'.format(key2))
		# print('type33',type(rslt_df))
		# print("w/ filter ", key2 ,rslt_df.shape[0])
		# print(rslt_df) 
		confunsion_dict.update({key2:rslt_df.shape[0]})
        
    # print(confunsion_dict)   
	return confunsion_dict,stain 

def calculate_all_values(confunsion_dict,Total_TN):

    all_val_cal_dict={"Precision":'',"Recall":'',"Specificity":'',"Negative_Prediction":'',"Accuracy":'',"F1":''}

    try:
        if confunsion_dict['TP'] != 0 or confunsion_dict['TN'] and Total_TN != 0:

            prescision=confunsion_dict['TP']/(confunsion_dict['TP']+confunsion_dict['FP'])
            recall=confunsion_dict['TP']/(confunsion_dict['TP']+confunsion_dict['FN'])
            
            specificity= (confunsion_dict['TN'] + Total_TN) /(confunsion_dict['TN']+confunsion_dict['FP']+Total_TN)
            negative_prediction=(confunsion_dict['TN'] +Total_TN)/(confunsion_dict['TN']+confunsion_dict['FN']+Total_TN)
            accuracy=(confunsion_dict['TP']+confunsion_dict['TN']+Total_TN)/(confunsion_dict['TP']+confunsion_dict['FP']+confunsion_dict['FN']+confunsion_dict['TN']+Total_TN)

            f1=2*prescision*recall/(prescision+recall)  
            all_val_cal_dict={"Precision":prescision,"Recall":recall,"Specificity":specificity,"Negative_Prediction":negative_prediction,"Accuracy":accuracy,"F1":f1}
            # print(all_val_cal_dict)
        else:
            pass
    except:
         confunsion_dict['TP'] or confunsion_dict['TN'] and Total_TN == 0

    return all_val_cal_dict

def create_csv_file_from_metric_allval(confunsion_dict,all_val_cal_dict):
    confusion_df=pd.DataFrame(columns=['Positive','Negative','Precision','Recall','Specificity','Negative_prediction','Accuracy','F1'],index=['Positive','Negative'])
    confusion_df.index.name='Actual/Prediction->'    
    confusion_df['Positive'] =[confunsion_dict['TP'],confunsion_dict['FP']]
    confusion_df['Negative'] =[confunsion_dict['FN'],confunsion_dict['TN']]
    confusion_df['Precision']=[all_val_cal_dict['Precision'],""]
    confusion_df['Recall']=[all_val_cal_dict['Recall'],""]
    confusion_df['Specificity']=[all_val_cal_dict['Specificity'],""]
    confusion_df['Negative_prediction']=[all_val_cal_dict['Negative_Prediction'],""]
    confusion_df['Accuracy']=[all_val_cal_dict['Accuracy'],""]
    confusion_df['F1']=[all_val_cal_dict['F1'],""]
    
    return confusion_df

if __name__ == "__main__":
    not_eligible = 0
    pathogen = 'FUNGUS'
    # pred_data = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/model_result_y.csv')
    # # pred_data = pred_data.sort_values('score').drop_duplicates(subset=['image_name', 'class_id'], keep='last') 
    # # pred_data.dropna(axis = 0, how ='any',inplace=True)
    # csv_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/input/csvs/190111-10-319-gms-20.csv'
    # data_n_svs = pd.read_csv(csv_path)
    # print('pred_data',pred_data.columns)
    # print('data_n_svs',data_n_svs.columns)
    # # x is data_n_svs , y is pred_data
    # final_df = check_pathogens(pred_data, data_n_svs)
    # confusion_mattrix
    final_df = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/data190111-10-319-gms-20_saved-model-march19-130_1.84.hdf5.csv')
    final_df=final_df[['image_name','PATHOGENS_x','PATHOGENS_y' ,'score' ]]
    confunsion_dict,stain = creation_of_confusion_matric(final_df, stain= pathogen)
    # metrics values
    # Extra_TN = any(final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) and final_df[ (final_df['PATHOGENS_x'] != pathogen) 
    #                                                                             & (~final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) ].dropna(axis = 0, how ='any')
    # # print('Total_TN', len(Extra_TN))
    # print('not_eligible', not_eligible)

    # Total_TN = len(Extra_TN['image_name']) + not_eligible
    Total_TN = 0
    all_val_cal_dict = calculate_all_values(confunsion_dict,Total_TN)
    # all_val_cal_dict = df_confusion_matric.calculate_all_values(confunsion_dict)
    # convert to csv file
    # csv_path = pred_csvs_path+'pred_csvs/{0}_{1}.csv'.format(basename,weight_model_name)
    # final_df.to_csv(csv_path)
    # prec_tp = unique_range_scores(csv_path,pathogen,others)
    # final_df = final_df.dropna(axis = 0, how ='any')
    print('confunsion_dict', confunsion_dict)
    print('Precision',all_val_cal_dict['Precision'])
    print('Recall',all_val_cal_dict['Recall'])
    print('Accuracy',all_val_cal_dict['Accuracy'] )
    print('F1',all_val_cal_dict['F1'])
    print('Specificity', all_val_cal_dict['Specificity'])
    print('Negative_prediction', all_val_cal_dict['Negative_Prediction'])



    