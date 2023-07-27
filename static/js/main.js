$(function() {

	'use strict';

	// Form

	var inputForm = function() {

		if ($('#inputForm').length > 0 ) {
			$("#inputForm").validate( {
				/* submit via ajax */
				submitHandler: function(form) {
					$.ajax({   	
				      type: "POST",
				      url: "/predict",
				      data: $(form).serialize(),
				      success: function(msg) {
						$('#dialog').html(msg);
						$('#dialog').dialog('open');
				       }
			      });    		
		  		}
				
				
			} );
		}
	};
	$(document).ready( function() {
		$('#dialog').dialog( {
			autoOpen: false,
			title: "Prediction Results",
			resizable: false,
			height: "auto",
			width: 400,
			modal: true,
			buttons: {
			  "Predict Another": function() {
				$( this ).dialog( "close" );
				$('#inputForm').trigger("reset");
			  }
			}
		 } );
	});
	inputForm();

});