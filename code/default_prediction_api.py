from flask import *
import pandas as pd
import predict_default
import xlrd

# creating the flask object
app = Flask(__name__)

REQUIRED_COLUMNS = ['account_amount_added_12_24m',
                    'age',
                    'merchant_category',
                    'merchant_group',
                    'has_paid',
                    'max_paid_inv_0_12m',
                    'name_in_email',
                    'num_active_inv',
                    'num_arch_dc_0_12m',
                    'num_arch_dc_12_24m',
                    'num_arch_ok_0_12m',
                    'num_arch_rem_0_12m',
                    'num_unpaid_bills',
                    'status_last_archived_0_24m',
                    'status_2nd_last_archived_0_24m',
                    'status_3rd_last_archived_0_24m',
                    'status_max_archived_0_6_months',
                    'status_max_archived_0_24_months',
                    'recovery_debt',
                    'sum_capital_paid_account_0_12m',
                    'sum_capital_paid_account_12_24m',
                    'sum_paid_inv_0_12m',
                    'time_hours']

# for each required columns we need to
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('upload_file.html')


ALLOWED_EXTENSIONS = {'xlsx'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/data', methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if allowed_file(file.filename):
                df = pd.read_excel(request.files.get('file'),  engine = 'openpyxl')
                for col in REQUIRED_COLUMNS:
                    if col not in df.columns:
                        return "One more required columns are missing. List of required columns are = {} ".format(REQUIRED_COLUMNS)

                # 2. Data prepartion
                df_final = predict_default.prepare_data(df)
                # 3. Prediction
                prediction = predict_default.predict(df_final)
                return prediction.to_html(header="true", table_id="table")

        else:
            return render_template('error_message.html')
    else:
        return render_template('error_message.html')




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)